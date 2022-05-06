import argparse
from enum import Enum
import time

import redis
import tensorflow as tf
import numpy as np
import lm_decoder

import neuralDecoder.singleStepRNN as singleStepRNN
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
from neuralDecoder.datasets.handwritingDataset import CHAR_DEF



class State(Enum):
    IDLE = 0
    DECODING = 1

def main():
    parser = argparse.ArgumentParser(description='Realtime RNN Decoder')
    parser.add_argument('--rnnDir', type=str, help='RNN model directory')
    parser.add_argument('--lmDir', type=str, help='LM model directory')
    parser.add_argument('--checkpointIdx', type=int, default=-1, help='Checkpoint index')
    parser.add_argument('--eosBlank', type=int, default=150, help='End of sentence blank')
    parser.add_argument('--inputDim', type=int, default=256, help='Input feature dimension')
    parser.add_argument('--bufferSize', type=int, default=10, help='Number of frames to buffer')
    parser.add_argument('--linearLayerIdx', type=int, default=0, help='Index of linear layer to load')
    parser.add_argument('--subsampleFactor', type=int, default=2, help='RNN 2nd layer subsample factor')
    parser.add_argument('--acousticScale', type=float, default=1.6, help='Acoustic scale')
    parser.add_argument('--debugStream', type=str, default=None, help='Debug data stream')
    parser.add_argument('--task', type=str, help='Task type', choices={"speech", "handwriting"})
    parser.add_argument('--blockMean', type=str, help='Block mean path')
    parser.add_argument('--blockStd', type=str, help='Block mean path')
    parser.add_argument('--simulateRealtime', action='store_true')
    args = parser.parse_args()

    debugStream = np.load(args.debugStream)
    assert len(debugStream.shape) == 2

    print(f"loading RNN from {args.rnnDir}")
    rnnModel, rnnArgs = singleStepRNN.loadRNN(args.rnnDir,
                                              checkpointIdx=args.checkpointIdx,
                                              loadLayerIdx=args.linearLayerIdx)

    if args.lmDir is not None:
        print(f"loading LM from {args.lmDir}")
        ngramDecoder = lmDecoderUtils.build_lm_decoder(
            args.lmDir,
            acoustic_scale=args.acousticScale,
            nbest=10
        )

    blockMean = None
    if args.blockMean is not None:
        print(f"loading block mean from {args.blockMean}")
        blockMean = np.load(args.blockMean).astype(np.float32)
    blockStd = None
    if args.blockStd is not None:
        print(f"loading block mean from {args.blockMean}")
        blockStd = np.load(args.blockStd).astype(np.float32)

    print("running...")

    state = State.IDLE
    blankClass = 0
    blankTicks = 0
    buffer = np.zeros([1, args.bufferSize, args.inputDim], dtype=np.float32)
    rnnStates = [rnnModel['rnnLayers'].initStates, None]
    frameCounter = 0
    allLogits = []
    while(1):
        if frameCounter >= debugStream.shape[0]:
            if state == State.DECODING:
                if args.lmDir is not None:
                    ngramDecoder.FinishDecoding()
                    for i in range(len(ngramDecoder.result())):
                        print(f'Final: {ngramDecoder.result()[i].sentence}')
                    ngramDecoder.Reset()
            break

        npBinnedNeural = debugStream[frameCounter]
        npBinnedNeural = npBinnedNeural.astype(np.float32)
        assert npBinnedNeural.shape == (args.inputDim,)
        if blockMean is not None:
            npBinnedNeural = (npBinnedNeural - blockMean) / blockStd
        npBinnedNeural = npBinnedNeural[np.newaxis, np.newaxis, :]
        # Append current frame to the tail of the buffer and remove the head
        buffer = np.concatenate([buffer[:, 1:, :], npBinnedNeural], axis=1)

        # Wait until buffer is full
        if frameCounter >= args.bufferSize and frameCounter % args.subsampleFactor == 0:
            logits, rnnStates = singleStepRNN.runSingleStep(rnnModel,
                                                            tf.constant(buffer, dtype=tf.float32),
                                                            rnnStates)
            #print(f"Single step RNN time: {time() - startT}")

            logits = logits.numpy()
            allLogits.append(logits)
            if args.task == 'handwriting':
                logits = lmDecoderUtils.rearrange_handwriting_logits(logits)
            elif args.task == 'speech':
                logits = lmDecoderUtils.rearrange_speech_logits(logits)

            if np.argmax(logits[..., :]) == blankClass:
                blankTicks += 1
            else:
                blankTicks = 0

            if blankTicks > args.eosBlank and state == State.DECODING:
                state = State.IDLE
                buffer = np.zeros([1, args.bufferSize, args.inputDim], dtype=np.float32)
                rnnStates = None
                frameCounter = -1
                print('End decoding')

                if args.lmDir is not None:
                    ngramDecoder.FinishDecoding()
                    for i in range(len(ngramDecoder.result())):
                        print(f'Final: {ngramDecoder.result()[i].sentence}')
                    ngramDecoder.Reset()
                else:
                    predictions = np.concatenate(allLogits, axis=1)
                    decodedStrings, _ = tf.nn.ctc_greedy_decoder(tf.transpose(predictions, [1, 0, 2]),
                                                                 [len(allLogits)],
                                                                 merge_repeated=True)
                    decoded = tf.sparse.to_dense(decodedStrings[0], default_value=-1).numpy()
                    if decoded.shape[1] > 0:
                        print(''.join([CHAR_DEF[i] for i in decoded[0]]))

                break

            if blankTicks == 1 and state == State.IDLE:
                state = State.DECODING
                print('Start decoding')

            if state == State.DECODING:
                if args.lmDir is None:
                    predictions = np.concatenate(allLogits, axis=1)
                    decodedStrings, _ = tf.nn.ctc_greedy_decoder(tf.transpose(predictions, [1, 0, 2]),
                                                                 [len(allLogits)],
                                                                 merge_repeated=True)
                    decoded = tf.sparse.to_dense(decodedStrings[0], default_value=-1).numpy()
                    if decoded.shape[1] > 0:
                        print(''.join([CHAR_DEF[i] for i in decoded[0]]))
                else:
                    lm_decoder.DecodeNumpy(ngramDecoder,logits[0]),
                    #print(f"Single step LM time: {time() - startT}")
                    if len(ngramDecoder.result()) > 0:
                        print(f'Partial: {ngramDecoder.result()[0].sentence}')

        frameCounter += 1
        if args.simulateRealtime:
            time.sleep(0.02)

if __name__ == "__main__":
    main()