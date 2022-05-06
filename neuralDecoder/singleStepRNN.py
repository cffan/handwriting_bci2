import os

from omegaconf import OmegaConf
import numpy as np
import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader
from scipy.ndimage.filters import gaussian_filter1d

import neuralDecoder.models as models


def loadRNN(modelDir, checkpointIdx=-1, loadLayerIdx=0):
    assert os.path.exists(modelDir)
    configPath = os.path.join(modelDir, 'args.yaml')
    assert os.path.exists(configPath)

    args = OmegaConf.load(configPath)

    # Build model
    model = models.GRU(args['model']['nUnits'],
                       args['model']['weightReg'],
                       args['model']['actReg'],
                       args['model']['subsampleFactor'],
                       args['dataset']['nClasses'] + 1,
                       args['model']['bidirectional'])
    model(tf.keras.Input(shape=(None, args['model']['inputLayerSize'])))
    model.trainable = False
    model.summary()

    # Input and norm layers
    normLayer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[args['dataset']['nInputFeatures']])
    linearLayer = tf.keras.layers.Dense(args['model']['inputLayerSize'])

    # Load weights
    if checkpointIdx >= 0:
        ckptPath = os.path.join(modelDir, f'ckpt-{checkpointIdx}')
    else:
        ckptPath = tf.train.latest_checkpoint(modelDir)

    if loadLayerIdx < 0:
        reader = py_checkpoint_reader.NewCheckpointReader(ckptPath)
        var_to_shape_map = reader.get_variable_to_shape_map()
        inputLayers = set()
        for layer in var_to_shape_map:
            if 'inputLayer_' in layer:
                inputLayers.add(layer.split('/')[0])
        print(f'Found {len(inputLayers)} input layers')
        loadLayerIdx = len(inputLayers) + loadLayerIdx

    ckptVars = {}
    ckptVars['net'] = model
    ckptVars['normLayer_'+str(loadLayerIdx)] = normLayer
    ckptVars['inputLayer_'+str(loadLayerIdx)] = linearLayer
    checkpoint = tf.train.Checkpoint(**ckptVars)
    checkpoint.restore(ckptPath).expect_partial()

    rnnModel = {}
    rnnModel['rnnLayers'] = model
    rnnModel['normLayer'] = normLayer
    rnnModel['linearInputLayer'] = linearLayer

    return rnnModel, args


@tf.function
def gaussSmooth(inputs, kernelSD=2, padding='SAME'):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.

    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """

    #get gaussian smoothing kernel
    inp = np.zeros([100], dtype=np.float32)
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel/np.sum(gaussKernel))

    # Apply depth_wise convolution
    B, T, C = inputs.shape.as_list()
    filters = tf.tile(gaussKernel[None, :, None, None], [1, 1, C, 1])  # [1, W, C, 1]
    inputs = inputs[:, None, :, :]  # [B, 1, T, C]
    smoothedInputs = tf.nn.depthwise_conv2d(inputs, filters, strides=[1, 1, 1, 1], padding=padding)
    smoothedInputs = tf.squeeze(smoothedInputs, 1)

    return smoothedInputs

@tf.function
def runSingleStep(rnn, x, states):
    x = rnn['normLayer'](x)
    x = gaussSmooth(x, padding='VALID')
    x = rnn['linearInputLayer'](x)
    logits, states = rnn['rnnLayers'](x, states, training=False, returnState=True)

    return logits, states
