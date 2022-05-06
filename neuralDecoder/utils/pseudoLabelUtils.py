import os
import pickle
import subprocess
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
from neuralDecoder.datasets.handwritingDataset import CHAR_DEF
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from neuralDecoder.datasets.handwritingDataset import HandwritingDataset


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _ints_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_to_ascii(text):
    return [ord(char) for char in text]

def charToId(char):
    return CHAR_DEF.index(char)

def writeTFRecords(transcriptions, recordPath):
    with tf.io.TFRecordWriter(recordPath) as writer:
        for trans in transcriptions:
            seqClassIDs = np.zeros([500,], dtype=np.int32)
            trans = trans.replace(' ', '>')
            trans = trans.replace('.', '~')
            seqClassIDs[:len(trans)] = [charToId(c) + 1 for c in trans]

            feature = {
                'seqClassIDs': _ints_feature(seqClassIDs),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def pseudo_label(ckptDir,
                 neuralDataDir,
                 outputDir,
                 session,
                 ngramDecoder,
                 gpt2Decoder,
                 gpt2Tokenizer,
                 labelPart='train',
                 labelTarget='gpt2',
                 startCkptIdx=None):
    cwd = os.getcwd()
    os.chdir(ckptDir)

    # Load model config and set which checkpoint to load
    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = './'
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = startCkptIdx
    args['dataset']['name'] = 'handwriting'

    # Initialize model
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    # Initialize validation dataset
    newValDataset = HandwritingDataset(os.path.join(neuralDataDir, session, labelPart),
                                       args['dataset']['nInputFeatures'],
                                       args['dataset']['nClasses'],
                                       args['dataset']['maxSeqElements'],
                                       args['dataset']['bufferSize'])
    newValDataset = newValDataset.build(args['batchSize'],
                                        isTraining=False)
    nsd.tfValDatasets = nsd.tfValDatasets[:-1] + [newValDataset]

    # Inference
    out = nsd.inference()

    print(f"raw cer {out['cer']}")

    if labelTarget == 'rnn':
        transcriptions = []
        for seq in out['decodedSeqs']:
            end = np.argmax(seq == -1)
            seq = seq[:end].tolist()
            seq = [CHAR_DEF[i] for i in seq]
            transcriptions.append(''.join(seq))
    elif labelTarget == 'gpt2':
        print(f"ngram cer {lmDecoderUtils.cer_with_lm_decoder(ngramDecoder, out)['cer']}")

        # Decode wiht LM model
        logits = out['logits']
        logitLengths = out['logitLengths']
        char_range = list(range(0, 26))
        symbol_range = [26, 27, 30, 29, 28]
        logits = logits[:, :, [31] + symbol_range + char_range]

        nbestOutputs = []
        for i in range(len(logits)):
            nbest = lmDecoderUtils.lm_decode(ngramDecoder,
                                             logits[i, :logitLengths[i]],
                                             returnNBest=True)
            nbestOutputs.append(nbest)

        gpt2Outputs = lmDecoderUtils.cer_with_gpt2_decoder(gpt2Decoder, gpt2Tokenizer, nbestOutputs, 1.0, out)
        print(f"gpt2 cer {gpt2Outputs['cer']}")
        transcriptions = gpt2Outputs['decoded_transcripts']

    tfRecordPath = os.path.join(
        outputDir,
        'label',
        session,
        'label.tfrecord'
    )
    path = Path(tfRecordPath)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(path)
    writeTFRecords(transcriptions, tfRecordPath)

    os.chdir(cwd)
    tf.compat.v1.reset_default_graph()


def generate_config(config_path, data_path, label_path, sessions, mru_sessions=10, **kwargs):
    total = len(sessions)
    conf = OmegaConf.load('../../../neuralDecoder/configs/dataset/handwriting_all_days_pseudo_label.yaml')
    conf['sessions'] = sessions
    conf['datasetToLayerMap'] = list(range(total))
    if total <= mru_sessions:
        conf['datasetProbability'] = [1.0 / total] * total
    else:
        conf['datasetProbability'] = [0.0] * (total - mru_sessions) + [1.0 / mru_sessions] * mru_sessions
    conf['datasetProbabilityVal'] = [1.0] if total == 1 else [0.0] * (total - 1) + [1.0]
    conf['dataDir'] = [str(data_path)] * total
    conf['labelDir'] = [None] * total if label_path is None else [None] + [str(label_path)] * (total - 1)
    conf['copyInputLayer'] = {} if total == 1 else {len(sessions) - 2: len(sessions) - 1}
    conf['syntheticMixingRate'] = 0.0

    for k, v in kwargs.items():
        if v is not None:
            conf[k] = v

    with config_path.open('w') as f:
        OmegaConf.save(conf, f)

def continuous_recalibration(output_dir, neural_data_dir, sessions, session_steps,
                             ngramDecoder=None,
                             gpt2Decoder=None,
                             gpt2Tokenizer=None,
                             labelTarget='rnn',
                             jobName='recal',
                             nBatchesToTrain=5000,
                             startCkptIdx=None,
                             subsetSize=None):
    state = {
        'current_session': 0
    }

    # Load state
    state_path = Path(output_dir, 'state.pkl')
    if state_path.exists():
        with state_path.open('rb') as f:
            state = pickle.load(f)

    sess_idx = state['current_session']
    while sess_idx < len(sessions):
        print(f"Running session {sessions[sess_idx]}")

        # Config
        lr = 0.01 if sess_idx == 0 else 0.001
        if startCkptIdx is not None and sess_idx == 0:
            exp_id = 'prev_exp'
        elif startCkptIdx is not None and sess_idx == 1:
            exp_id = f"dataset=handwriting_pseudo_label_{sessions[sess_idx]},learnRateStart={lr},loadCheckpointIdx={startCkptIdx},model.dropout=0.8,nBatchesToTrain={nBatchesToTrain}"
        else:
            exp_id = f"dataset=handwriting_pseudo_label_{sessions[sess_idx]},learnRateStart={lr},model.dropout=0.8,nBatchesToTrain={nBatchesToTrain}"
        config_dir = Path(output_dir, 'config', sessions[sess_idx], 'dataset')
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir.joinpath(f"handwriting_pseudo_label_{sessions[sess_idx]}.yaml")

        if not config_path.exists():
            data_path = Path(neural_data_dir)
            if sess_idx == 0 or labelTarget is None:
                label_path = None
            else:
                label_path = Path(output_dir, 'label')

            generate_config(config_path,
                            data_path,
                            label_path,
                            sessions[:sess_idx + 1],
                            mru_sessions=10,
                            subsetSize=subsetSize)

        output_path = Path(output_dir, 'checkpoint', sessions[sess_idx])
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Train
        if session_steps[sessions[sess_idx]][0] == 1:
            log_dir = Path(output_dir, 'log', sessions[sess_idx])
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir.joinpath('out.txt')
            stderr_path = log_dir.joinpath('err.txt')
            stdout_file = stdout_path.open('a')
            stderr_file = stderr_path.open('a')

            cmd = ['python',
                   '-m', 'neuralDecoder.main',
                   '--config-dir', Path(output_dir, 'config', sessions[sess_idx]),
                   '--multirun', 'hydra/launcher=gpu_slurm',
                   f'hydra.job.name={jobName}_{sessions[sess_idx]}',
                   f'dataset=handwriting_pseudo_label_{sessions[sess_idx]}',
                   'model.dropout=0.8',
                   f'learnRateStart={lr}',
                   f'nBatchesToTrain={nBatchesToTrain}',
                   f'outputDir={str(output_path)}',
            ]

            # Specify starting checkpoint for a pretrained model
            if sess_idx == 1 and startCkptIdx is not None:
                cmd.append(f'loadCheckpointIdx={startCkptIdx}')

            # Load previous checkpoint
            if sess_idx > 0:
                prev_lr = 0.01 if sess_idx == 1 else lr
                if startCkptIdx is not None and sess_idx == 1:
                    prev_exp_id = 'prev_exp'
                elif startCkptIdx is not None and sess_idx == 2:
                    prev_exp_id = f"dataset=handwriting_pseudo_label_{sessions[sess_idx - 1]},learnRateStart={prev_lr},loadCheckpointIdx={startCkptIdx},model.dropout=0.8,nBatchesToTrain={nBatchesToTrain}"
                else:
                    prev_exp_id = f"dataset=handwriting_pseudo_label_{sessions[sess_idx - 1]},learnRateStart={prev_lr},model.dropout=0.8,nBatchesToTrain={nBatchesToTrain}"
                load_path = Path(output_dir, 'checkpoint', sessions[sess_idx - 1], prev_exp_id)
                escaped_load_path = str(load_path).replace('=', '\=').replace(',', '\,')
                cmd += ['loadDir=' + escaped_load_path]


            process = subprocess.run(cmd,
                                     stdout=stdout_file,
                                     stderr=stderr_file)
            stdout_file.close()
            stderr_file.close()

            if process.returncode != 0:
                print(f'ERROR: return code {process.returncode}')
                break

        # Evaluate
        # TODO

        # Label
        if session_steps[sessions[sess_idx]][2] == 1:
            if sess_idx < len(sessions) - 1:
                pseudo_label(
                    str(Path(output_dir, 'checkpoint', sessions[sess_idx], exp_id)),
                    neural_data_dir,
                    output_dir,
                    sessions[sess_idx + 1],
                    ngramDecoder,
                    gpt2Decoder,
                    gpt2Tokenizer,
                    'train',
                    labelTarget,
                    startCkptIdx if sess_idx == 0 else None
                )

        sess_idx += 1
        state['current_session'] = sess_idx
        with state_path.open('wb') as f:
            pickle.dump(state, f)


def extract_tfrecord_metric(path, metric_name):
    event_acc = EventAccumulator(path, size_guidance={'tensors': 0})
    event_acc.Reload()

    metrics = pd.DataFrame([(s, tf.make_ndarray(t).item()) for w, s, t in event_acc.Tensors(metric_name)],
                             columns=['step', metric_name])

    return metrics

def inference(ckptDir, ckptIdx, valDir):

    # Change working dir to checkpoint dir
    cwd = os.getcwd()
    os.chdir(ckptDir)

    # Load model config and set which checkpoint to load
    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = './'
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = ckptIdx
    args['dataset']['name'] = 'handwriting'

    # Initialize model
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    # Initialize validation dataset
    if valDir is not None:
        newValDataset = HandwritingDataset(valDir,
                                           args['dataset']['nInputFeatures'],
                                           args['dataset']['nClasses'],
                                           args['dataset']['maxSeqElements'],
                                           args['dataset']['bufferSize'])
        newValDataset = newValDataset.build(args['batchSize'],
                                            isTraining=False)
        nsd.tfValDatasets = [newValDataset]

    # Inference
    out = nsd.inference()

    os.chdir(cwd)

    return out