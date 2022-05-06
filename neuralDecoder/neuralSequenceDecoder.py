import os
import copy
import random
from datetime import datetime

import numpy as np
import scipy.io
import scipy.special
import tensorflow as tf
from omegaconf import OmegaConf

import neuralDecoder.lrSchedule as lrSchedule
import neuralDecoder.models as models
from neuralDecoder.datasets import getDataset
from neuralDecoder.singleStepRNN import gaussSmooth


class NeuralSequenceDecoder(object):
    """
    This class encapsulates all the functionality needed for training, loading and running the neural sequence decoder RNN.
    To use it, initialize this class and then call .train() or .inference(). It can also be run from the command line (see bottom
    of the script). The args dictionary passed during initialization is used to configure all aspects of its behavior.
    """

    def __init__(self, args):
        self.args = args

        if not os.path.isdir(self.args['outputDir']):
            os.mkdir(self.args['outputDir'])

        #record these parameters
        if self.args['mode'] == 'train':
            with open(os.path.join(args['outputDir'], 'args.yaml'), 'w') as f:
                OmegaConf.save(config=self.args, f=f)

        #random variable seeding
        if self.args['seed'] == -1:
            self.args['seed'] = datetime.now().microsecond
        np.random.seed(self.args['seed'])
        tf.random.set_seed(self.args['seed'])
        random.seed(self.args['seed'])

        # Init GRU model
        self.model = models.GRU(self.args['model']['nUnits'],
                         self.args['model']['weightReg'],
                         self.args['model']['actReg'],
                         self.args['model']['subsampleFactor'],
                         self.args['dataset']['nClasses'] + 1,
                         self.args['model']['bidirectional'],
                         self.args['model']['dropout'])
        self.model(tf.keras.Input(shape=(None, self.args['model']['inputLayerSize'])))
        self.model.trainable = self.args['model'].get('trainable', True)
        self.model.summary()

        self._prepareForTraining()

    def _buildInputLayers(self, isTraining):
        # Build day transformation and normalization layers
        self.nInputLayers = np.max(self.args['dataset']['datasetToLayerMap'])+1
        self.inputLayers = []
        self.normLayers = []
        for layerIdx in range(self.nInputLayers):
            datasetIdx = np.argwhere(
                np.array(self.args['dataset']['datasetToLayerMap']) == layerIdx)
            datasetIdx = datasetIdx[0, 0]

            nInputFeatures = self.args['dataset']['nInputFeatures']

            # Adapt normalization layer with all data.
            normLayer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[
                                                                                 nInputFeatures])
            if isTraining and self.args['normLayer']:
                normLayer.adapt(self.tfAdaptDatasets[datasetIdx].take(-1))

            if self.args['model']['inputLayerSize'] == nInputFeatures:
                kernelInit = tf.keras.initializers.identity()
            else:
                kernelInit = 'glorot_uniform'
            linearLayer = tf.keras.layers.Dense(self.args['model']['inputLayerSize'],
                                                kernel_initializer=kernelInit,
                                                kernel_regularizer=tf.keras.regularizers.L2(self.args['model']['weightReg']))
            linearLayer.build(input_shape=[nInputFeatures])

            self.inputLayers.append(linearLayer)
            self.normLayers.append(normLayer)

    def _buildOptimizer(self):
        #define the gradient descent optimizer
        if self.args['warmUpSteps'] > 0:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.args['learnRateStart'],
                decay_steps=self.args.get('learnRateDecaySteps', self.args['nBatchesToTrain']) - self.args['warmUpSteps'],
                end_learning_rate=self.args['learnRateEnd'],
                power=1.0,
            )
            learning_rate_fn = lrSchedule.WarmUp(
                initial_learning_rate=self.args['learnRateStart'],
                decay_schedule_fn=lr_schedule,
                warmup_steps=self.args['warmUpSteps']
            )
        else:
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(self.args['learnRateStart'],
                                                                             self.args.get('learnRateDecaySteps', self.args['nBatchesToTrain']),
                                                                             end_learning_rate=self.args['learnRateEnd'],
                                                                             power=1.0, cycle=False, name=None)

        self.optimizer = tf.keras.optimizers.Adam(
            beta_1=0.9, beta_2=0.999, epsilon=1e-01, learning_rate=learning_rate_fn)

    def _prepareForTraining(self):
        #build the dataset pipelines
        self.tfAdaptDatasets = []
        self.tfTrainDatasets = []
        self.tfValDatasets = []
        for i, (thisDataset, thisDataDir) in enumerate(zip(self.args['dataset']['sessions'], self.args['dataset']['dataDir'])):
            trainDir = os.path.join(thisDataDir, thisDataset, 'train')
            syntheticDataDir = None
            if (self.args['dataset']['syntheticMixingRate'] > 0 and
                self.args['dataset']['syntheticDataDir'] is not None):
                syntheticDataDir = os.path.join(self.args['dataset']['syntheticDataDir'],
                                                f'{thisDataset}_syntheticSentences')

            datasetName = self.args['dataset']['name']
            labelDir = None
            labelDirs = self.args['dataset'].get('labelDir', None)
            if labelDirs is not None and labelDirs[i] is not None:
                labelDir = os.path.join(labelDirs[i], thisDataset)
            newTrainDataset = getDataset(datasetName)(trainDir,
                                                      self.args['dataset']['nInputFeatures'],
                                                      self.args['dataset']['nClasses'],
                                                      self.args['dataset']['maxSeqElements'],
                                                      self.args['dataset']['bufferSize'],
                                                      syntheticDataDir,
                                                      self.args['dataset']['syntheticMixingRate'],
                                                      self.args['dataset'].get('subsetSize', -1),
                                                      labelDir)
            newTrainDataset, newDatasetForAdapt = newTrainDataset.build(
                self.args['batchSize'],
                isTraining=True)

            valDir = os.path.join(thisDataDir, thisDataset, 'test')
            newValDataset = getDataset(datasetName)(valDir,
                                                    self.args['dataset']['nInputFeatures'],
                                                    self.args['dataset']['nClasses'],
                                                    self.args['dataset']['maxSeqElements'],
                                                    self.args['dataset']['bufferSize'])
            newValDataset = newValDataset.build(self.args['batchSize'],
                                                isTraining=False)

            self.tfAdaptDatasets.append(newDatasetForAdapt)
            self.tfTrainDatasets.append(newTrainDataset)
            self.tfValDatasets.append(newValDataset)

        # Define input layers, including feature normalization which is adapted on the training data
        self._buildInputLayers(isTraining=True)

        # Train dataset selector. Used for switch between different day's data during training.
        self.trainDatasetSelector = {}
        self.trainDatasetIterators = [iter(d) for d in self.tfTrainDatasets]
        for x in range(len(self.args['dataset']['sessions'])):
            self.trainDatasetSelector[x] = lambda x=x: self._datasetLayerTransform(self.trainDatasetIterators[x].get_next(),
                                                                                   self.normLayers[self.args['dataset']['datasetToLayerMap'][x]],
                                                                                   self.args['dataset']['whiteNoiseSD'],
                                                                                   self.args['dataset']['constantOffsetSD'],
                                                                                   self.args['dataset']['randomWalkSD'],
                                                                                   self.args['dataset']['staticGainSD'])

        self._buildOptimizer()

        #define a list of all trainable variables for optimization
        self.trainableVariables = []
        if self.args['trainableBackend']:
            self.trainableVariables.extend(self.model.trainable_variables)

        if self.args['trainableInput']:
            for x in range(len(self.inputLayers)):
                self.trainableVariables.extend(
                    self.inputLayers[x].trainable_variables)

        #clear old checkpoints
        #ckptFiles = [str(x) for x in pathlib.Path(self.args['outputDir']).glob("ckpt-*")]
        #for file in ckptFiles:
        #    os.remove(file)

        #if os.path.isfile(self.args['outputDir'] + '/checkpoint'):
        #    os.remove(self.args['outputDir'] + '/checkpoint')

        #saving/loading
        ckptVars = {}
        ckptVars['net'] = self.model
        for x in range(len(self.normLayers)):
            ckptVars['normLayer_'+str(x)] = self.normLayers[x]
            ckptVars['inputLayer_'+str(x)] = self.inputLayers[x]

        # Resume if checkpoint exists in outputDir
        resume = os.path.exists(os.path.join(self.args['outputDir'], 'checkpoint'))
        if resume:
            # Resume training, so we need to load optimizer and step from checkpoint.
            ckptVars['step'] = tf.Variable(0)
            ckptVars['bestValCer'] = tf.Variable(1.0)
            ckptVars['optimizer'] = self.optimizer
            self.checkpoint = tf.train.Checkpoint(**ckptVars)
            ckptPath = tf.train.latest_checkpoint(self.args['outputDir'])
            # If in infer mode, we may want to load a particular checkpoint idx
            if self.args['mode'] == 'infer':
                if self.args['loadCheckpointIdx'] is not None:
                    ckptPath = os.path.join(self.args['outputDir'], f'ckpt-{self.args["loadCheckpointIdx"]}')
            print('Loading from : ' + ckptPath)
            self.checkpoint.restore(ckptPath).expect_partial()
        else:
            if self.args['loadDir'] != None and os.path.exists(os.path.join(self.args['loadDir'], 'checkpoint')):
                if self.args['loadCheckpointIdx'] is not None:
                    ckptPath = os.path.join(self.args['loadDir'], f'ckpt-{self.args["loadCheckpointIdx"]}')
                else:
                    ckptPath = tf.train.latest_checkpoint(self.args['loadDir'])

                print('Loading from : ' + ckptPath)
                self.checkpoint = tf.train.Checkpoint(**ckptVars)
                self.checkpoint.restore(ckptPath)

                if 'copyInputLayer' in self.args['dataset'] and self.args['dataset']['copyInputLayer'] is not None:
                    for f, t in self.args['dataset']['copyInputLayer'].items():
                       for vf, vt in zip(self.inputLayers[int(f)].variables, self.inputLayers[int(t)].variables):
                           vt.assign(vf)

                # After loading, we need to put optimizer and step back to checkpoint in order to save them.
                ckptVars['step'] = tf.Variable(0)
                ckptVars['bestValCer'] = tf.Variable(1.0)
                ckptVars['optimizer'] = self.optimizer
                self.checkpoint = tf.train.Checkpoint(**ckptVars)
            else:
                # Nothing to load.
                ckptVars['step'] = tf.Variable(0)
                ckptVars['bestValCer'] = tf.Variable(1.0)
                ckptVars['optimizer'] = self.optimizer
                self.checkpoint = tf.train.Checkpoint(**ckptVars)

        self.ckptManager = tf.train.CheckpointManager(
            self.checkpoint, self.args['outputDir'], max_to_keep=None if self.args['batchesPerSave'] > 0 else 10)

        # Tensorboard summary
        if self.args['mode'] == 'train':
            self.summary_writer = tf.summary.create_file_writer(
                self.args['outputDir'])

    def _datasetLayerTransform(self, dat, normLayer, whiteNoiseSD, constantOffsetSD, randomWalkSD, staticGainSD):
        features = dat['inputFeatures']
        features = normLayer(dat['inputFeatures'])

        featShape = tf.shape(features)
        batchSize = featShape[0]
        featDim = featShape[2]
        if staticGainSD > 0:
            warpMat = tf.tile(tf.eye(dat['inputFeatures'].shape[2])[
                              tf.newaxis, :, :], [batchSize, 1, 1])
            warpMat += tf.random.normal(tf.shape(warpMat),
                                        mean=0, stddev=staticGainSD)
            features = tf.linalg.matmul(features, warpMat)

        if whiteNoiseSD > 0:
            features += tf.random.normal(featShape, mean=0, stddev=whiteNoiseSD)

        if constantOffsetSD > 0:
            features += tf.random.normal([batchSize, 1, featDim], mean=0,
                                         stddev=constantOffsetSD)

        if randomWalkSD > 0:
            features += tf.math.cumsum(tf.random.normal(
                featShape, mean=0, stddev=randomWalkSD), axis=self.args['randomWalkAxis'])

        if self.args['smoothInputs']:
            features = gaussSmooth(
                features, kernelSD=self.args['smoothKernelSD'])

        outDict = {'inputFeatures': features,
                   #'classLabelsOneHot': dat['classLabelsOneHot'],
                   'newClassSignal': dat['newClassSignal'],
                   'seqClassIDs': dat['seqClassIDs'],
                   'nTimeSteps': dat['nTimeSteps'],
                   'nSeqElements': dat['nSeqElements'],
                   'ceMask': dat['ceMask'],
                   'transcription': dat['transcription']}

        return outDict

    def train(self):
        perBatchData_train = np.zeros([self.args['nBatchesToTrain'] + 1, 6])
        perBatchData_val = np.zeros([self.args['nBatchesToTrain'] + 1, 6])

        # Restore snapshot
        restoredStep = int(self.checkpoint.step)
        if restoredStep > 0:
            outputSnapshot = scipy.io.loadmat(self.args['outputDir']+'/outputSnapshot')
            perBatchData_train = outputSnapshot['perBatchData_train']
            perBatchData_val = outputSnapshot['perBatchData_val']

        saveBestCheckpoint = self.args['batchesPerSave'] == 0
        bestValCer = self.checkpoint.bestValCer
        print('bestValCer: ' + str(bestValCer))
        for batchIdx in range(restoredStep, self.args['nBatchesToTrain'] + 1):
            #--training--
            datasetIdx = int(np.argwhere(
                np.random.multinomial(1, self.args['dataset']['datasetProbability']))[0][0])
            layerIdx = self.args['dataset']['datasetToLayerMap'][datasetIdx]

            dtStart = datetime.now()
            trainOut = self._trainStep(
                tf.constant(datasetIdx, dtype=tf.int32),
                tf.constant(layerIdx, dtype=tf.int32))

            self.checkpoint.step.assign_add(1)
            totalSeconds = (datetime.now()-dtStart).total_seconds()
            self._addRowToStatsTable(
                perBatchData_train, batchIdx, totalSeconds, trainOut, True)
            print(f'Train batch {batchIdx}: ' +
                  f'loss: {(trainOut["predictionLoss"] + trainOut["regularizationLoss"]):.2f} ' +
                  f'gradNorm: {trainOut["gradNorm"]:.2f} ' +
                  f'time {totalSeconds:.2f}')

            #--validation--
            if batchIdx % self.args['batchesPerVal'] == 0:
                dtStart = datetime.now()
                valOutputs = self.inference()
                totalSeconds = (datetime.now()-dtStart).total_seconds()

                valOutputs['seqErrorRate'] = float(
                    np.sum(valOutputs['editDistances'])) / np.sum(valOutputs['trueSeqLengths'])
                self._addRowToStatsTable(
                    perBatchData_val, batchIdx, totalSeconds, valOutputs, False)
                print(f'Val batch {batchIdx}: ' +
                      f'CER: {valOutputs["seqErrorRate"]:.2f} ' +
                      f'time {totalSeconds:.2f}')

                if saveBestCheckpoint and valOutputs['seqErrorRate'] < bestValCer:
                    bestValCer = valOutputs['seqErrorRate']
                    self.checkpoint.bestValCer.assign(bestValCer)
                    savedCkpt = self.ckptManager.save(checkpoint_number=batchIdx)
                    print(f'Checkpoint saved {savedCkpt}')

                #save a snapshot of key RNN outputs/variables so an outside program can plot them if desired
                outputSnapshot = {}
                outputSnapshot['logitsSnapshot'] = trainOut['logits'][0, :, :].numpy()
                outputSnapshot['rnnUnitsSnapshot'] = trainOut['rnnUnits'][0, :, :].numpy(
                )
                outputSnapshot['inputFeaturesSnapshot'] = trainOut['inputFeatures'][0, :, :].numpy(
                )
                #outputSnapshot['classLabelsSnapshot'] = trainOut['classLabels'][0, :, :].numpy(
                #)
                outputSnapshot['perBatchData_train'] = perBatchData_train
                outputSnapshot['perBatchData_val'] = perBatchData_val
                outputSnapshot['seqIDs'] = trainOut['seqIDs'][0, :].numpy()
                scipy.io.savemat(
                    self.args['outputDir']+'/outputSnapshot', outputSnapshot)

            if self.args['batchesPerSave'] > 0 and batchIdx % self.args['batchesPerSave'] == 0:
                savedCkpt = self.ckptManager.save(checkpoint_number=batchIdx)
                print(f'Checkpoint saved {savedCkpt}')

    def inference(self, returnData=False):
        #run through the specified dataset a single time and return the outputs
        infOut = {}
        infOut['logits'] = []
        infOut['logitLengths'] = []
        infOut['decodedSeqs'] = []
        infOut['editDistances'] = []
        infOut['trueSeqLengths'] = []
        infOut['trueSeqs'] = []
        infOut['transcriptions'] = []
        allData = []

        for datasetIdx, valProb in enumerate(self.args['dataset']['datasetProbabilityVal']):
            if valProb <= 0:
                continue

            layerIdx = self.args['dataset']['datasetToLayerMap'][datasetIdx]

            for data in self.tfValDatasets[datasetIdx]:
                out = self._valStep(data, layerIdx)

                infOut['logits'].append(out['logits'].numpy())
                infOut['editDistances'].append(out['editDistance'].numpy())
                infOut['trueSeqLengths'].append(out['nSeqElements'].numpy())
                infOut['logitLengths'].append(out['logitLengths'].numpy())

                infOut['trueSeqs'].append(out['trueSeq'].numpy()-1)

                tmp = tf.sparse.to_dense(
                    out['decodedStrings'][0], default_value=-1).numpy()
                paddedMat = np.zeros(
                    [tmp.shape[0], self.args['dataset']['maxSeqElements']]).astype(np.int32)-1
                end = min(tmp.shape[1], self.args['dataset']['maxSeqElements'])
                paddedMat[:, :end] = tmp[:, :end]
                infOut['decodedSeqs'].append(paddedMat)

                infOut['transcriptions'].append(out['transcription'].numpy())

                if returnData:
                    allData.append(data)

        # Logits have different length
        infOut['logits'] = [l for batch in infOut['logits'] for l in list(batch)]
        maxLogitLength = max([l.shape[0] for l in infOut['logits']])
        infOut['logits'] = [np.pad(l, [[0, maxLogitLength-l.shape[0]], [0, 0]]) for l in infOut['logits']]
        infOut['logits'] = np.stack(infOut['logits'], axis=0)

        infOut['logitLengths'] = np.concatenate(infOut['logitLengths'], axis=0)
        infOut['decodedSeqs'] = np.concatenate(infOut['decodedSeqs'], axis=0)
        infOut['editDistances'] = np.concatenate(
            infOut['editDistances'], axis=0)
        infOut['trueSeqLengths'] = np.concatenate(
            infOut['trueSeqLengths'], axis=0)
        infOut['trueSeqs'] = np.concatenate(infOut['trueSeqs'], axis=0)
        infOut['transcriptions'] = np.concatenate(
            infOut['transcriptions'], axis=0)

        infOut['cer'] = np.sum(infOut['editDistances']) / float(np.sum(infOut['trueSeqLengths']))

        if returnData:
            return infOut, allData
        else:
            return infOut

    def _addRowToStatsTable(self, currentTable, batchIdx, computationTime, minibatchOutput, isTrainBatch):
        currentTable[batchIdx, :] = np.array([batchIdx,
                                              computationTime,
                                              minibatchOutput['predictionLoss'] if isTrainBatch else 0.0,
                                              minibatchOutput['regularizationLoss'] if isTrainBatch else 0.0,
                                              minibatchOutput['seqErrorRate'],
                                              minibatchOutput['gradNorm'] if isTrainBatch else 0.0])

        prefix = 'train' if isTrainBatch else 'val'
        with self.summary_writer.as_default():
            if isTrainBatch:
                tf.summary.scalar(
                    f'{prefix}/predictionLoss', minibatchOutput['predictionLoss'], step=batchIdx)
                tf.summary.scalar(
                    f'{prefix}/regLoss', minibatchOutput['regularizationLoss'], step=batchIdx)
                tf.summary.scalar(f'{prefix}/gradNorm',
                                  minibatchOutput['gradNorm'], step=batchIdx)
            tf.summary.scalar(f'{prefix}/seqErrorRate',
                              minibatchOutput['seqErrorRate'], step=batchIdx)
            tf.summary.scalar(f'{prefix}/computationTime',
                              computationTime, step=batchIdx)
            if isTrainBatch:
                tf.summary.scalar(
                    f'{prefix}/lr', self.optimizer._decayed_lr(tf.float32), step=batchIdx)

    @tf.function()
    def _trainStep(self, datasetIdx, layerIdx):
        #loss function & regularization
        data = tf.switch_case(datasetIdx, self.trainDatasetSelector)

        inputTransformSelector = {}
        for x in range(self.nInputLayers):
            inputTransformSelector[x] = lambda x=x: self.inputLayers[x](
                data['inputFeatures'], training=True)

        regLossSelector = {}
        for x in range(self.nInputLayers):
            regLossSelector[x] = lambda x=x: self.inputLayers[x].losses

        with tf.GradientTape() as tape:
            inputTransformedFeatures = tf.switch_case(
                layerIdx, inputTransformSelector)
            predictions = self.model(inputTransformedFeatures, training=True)
            regularization_loss = tf.math.add_n(self.model.losses) + \
                tf.math.add_n(tf.switch_case(layerIdx, regLossSelector))

            batchSize = tf.shape(data['inputFeatures'])[0]
            if self.args['lossType'] == 'ctc':
                sparseLabels = tf.cast(tf.sparse.from_dense(
                    data['seqClassIDs']), dtype=tf.int32)
                sparseLabels = tf.sparse.SparseTensor(
                    indices=sparseLabels.indices,
                    values=sparseLabels.values-1,
                    dense_shape=[batchSize, self.args['dataset']['maxSeqElements']])

                nTimeSteps = tf.cast(data['nTimeSteps']/self.args['model']['subsampleFactor'], dtype=tf.int32)

                pred_loss = tf.compat.v1.nn.ctc_loss_v2(sparseLabels,
                                                        predictions,
                                                        None,
                                                        nTimeSteps,
                                                        logits_time_major=False,
                                                        unique=None,
                                                        blank_index=-1,
                                                        name=None)

                pred_loss = tf.reduce_mean(pred_loss)

            elif self.args['lossType'] == 'ce':
                mask = tf.tile(data['ceMask'][:, :, tf.newaxis], [
                               1, 1, self.args['dataset']['nClasses']])
                ceLoss = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True)
                pred_loss = ceLoss(
                    data['classLabelsOneHot'], predictions[:, :, 0:-1]*mask)

            total_loss = pred_loss + regularization_loss

        #compute gradients + clip
        grads = tape.gradient(total_loss, self.trainableVariables)
        grads, gradNorm = tf.clip_by_global_norm(
            grads, self.args['gradClipValue'])

        #only apply if gradients are finite and we are in train mode
        allIsFinite = []
        for g in grads:
            if g != None:
                allIsFinite.append(tf.reduce_all(tf.math.is_finite(g)))
        gradIsFinite = tf.reduce_all(tf.stack(allIsFinite))

        if gradIsFinite:
            self.optimizer.apply_gradients(zip(grads, self.trainableVariables))

        #compute sequence-element error rate (edit distance) if we are in validation & ctc mode
        #return interval activations so we can visualize what's going on
        intermediate_output = self.model.getIntermediateLayerOutput(
            inputTransformedFeatures)

        output = {}
        output['logits'] = predictions
        output['rnnUnits'] = intermediate_output
        output['inputFeatures'] = data['inputFeatures']
        #output['classLabels'] = data['classLabelsOneHot']
        output['predictionLoss'] = pred_loss
        output['regularizationLoss'] = regularization_loss
        output['gradNorm'] = gradNorm
        output['seqIDs'] = data['seqClassIDs']
        output['seqErrorRate'] = tf.constant(0.0)

        return output

    def _valStep(self, data, layerIdx):
        data = self._datasetLayerTransform(
            data, self.normLayers[layerIdx], 0, 0, 0, 0)

        inputTransformedFeatures = self.inputLayers[layerIdx](
            data['inputFeatures'], training=False)
        predictions = self.model(inputTransformedFeatures, training=False)

        batchSize = tf.shape(data['seqClassIDs'])[0]
        if self.args['lossType'] == 'ctc':
            sparseLabels = tf.cast(tf.sparse.from_dense(
                data['seqClassIDs']), dtype=tf.int32)
            sparseLabels = tf.sparse.SparseTensor(
                indices=sparseLabels.indices,
                values=sparseLabels.values-1,
                dense_shape=[batchSize, self.args['dataset']['maxSeqElements']])

            nTimeSteps = tf.cast(
                data['nTimeSteps'] / self.args['model']['subsampleFactor'], dtype=tf.int32)

            pred_loss = tf.compat.v1.nn.ctc_loss_v2(sparseLabels, predictions,
                                                    tf.cast(
                                                        data['nSeqElements'], dtype=tf.int32), nTimeSteps,
                                                    logits_time_major=False, unique=None, blank_index=-1, name=None)

            pred_loss = tf.reduce_mean(pred_loss)

        elif self.args['lossType'] == 'ce':
            mask = tf.tile(data['ceMask'][:, :, tf.newaxis], [
                           1, 1, self.args['dataset']['nClasses']])
            ceLoss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            pred_loss = ceLoss(data['classLabelsOneHot'],
                               predictions[:, :, 0:-1]*mask)

        if self.args['lossType'] == 'ctc':
            sparseLabels = tf.cast(tf.sparse.from_dense(
                data['seqClassIDs']), dtype=tf.int32)
            sparseLabels = tf.sparse.SparseTensor(
                indices=sparseLabels.indices,
                values=sparseLabels.values-1,
                dense_shape=[batchSize, self.args['dataset']['maxSeqElements']])

            decodedStrings, _ = tf.nn.ctc_greedy_decoder(tf.transpose(predictions, [1, 0, 2]),
                                                         nTimeSteps,
                                                         merge_repeated=True)
            editDistance = tf.edit_distance(decodedStrings[0], tf.cast(
                sparseLabels, tf.int64), normalize=False)
            seqErrorRate = tf.cast(tf.reduce_sum(editDistance), dtype=tf.float32)/tf.cast(
                tf.reduce_sum(data['nSeqElements']), dtype=tf.float32)
        else:
            decodedStrings = [tf.sparse.SparseTensor(
                indices=[[0, 0]], values=[tf.constant(1, dtype=tf.int64)], dense_shape=[2, 2])]
            seqErrorRate = 0.0

        output = {}
        output['logits'] = predictions
        output['decodedStrings'] = decodedStrings
        output['seqErrorRate'] = seqErrorRate
        output['editDistance'] = editDistance
        output['trueSeq'] = data['seqClassIDs']
        output['nSeqElements'] = data['nSeqElements']
        output['transcription'] = data['transcription']
        output['logitLengths'] = nTimeSteps

        return output


def timeWarpDataElement(dat, timeScalingRange):
    warpDat = {}
    warpDat['seqClassIDs'] = dat['seqClassIDs']
    warpDat['nSeqElements'] = dat['nSeqElements']
    warpDat['transcription'] = dat['transcription']

    #nTimeSteps, inputFeatures need to be modified
    globalTimeFactor = 1 + \
        (tf.random.uniform(shape=(), dtype=tf.float32)-0.5)*timeScalingRange
    warpDat['nTimeSteps'] = tf.cast(
        tf.cast(dat['nTimeSteps'], dtype=tf.float32)*globalTimeFactor, dtype=tf.int64)

    newIdx = tf.linspace(0.0, tf.cast(
        dat['nTimeSteps']-1, dtype=tf.float32), warpDat['nTimeSteps'])
    newIdx = tf.cast(newIdx, dtype=tf.int32)

    warpDat['inputFeatures'] = tf.gather(dat['inputFeatures'], newIdx, axis=0)
    warpDat['classLabelsOneHot'] = tf.gather(
        dat['classLabelsOneHot'], newIdx, axis=0)
    warpDat['newClassSignal'] = tf.gather(
        dat['newClassSignal'], newIdx, axis=0)
    warpDat['ceMask'] = tf.gather(dat['ceMask'], newIdx, axis=0)

    return warpDat
