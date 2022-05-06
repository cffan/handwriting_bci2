import argparse
from enum import Enum
from time import time

import redis
import tensorflow as tf
import numpy as np
import lm_decoder

import neuralDecoder.singleStepRNN as singleStepRNN
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils


class State(Enum):
    IDLE = 0
    DECODING = 1

def main():
    parser = argparse.ArgumentParser(description='Realtime RNN Decoder')
    parser.add_argument('--host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--db', type=int, default=0, help='Redis db')
    parser.add_argument('--rnnDir', type=str, help='RNN model directory')
    parser.add_argument('--lmDir', type=str, help='LM model directory')
    parser.add_argument('--checkpointIdx', type=int, default=-1, help='Checkpoint index')
    parser.add_argument('--eosBlank', type=int, default=75, help='End of sentence blank')
    parser.add_argument('--inputDim', type=int, default=256, help='Input feature dimension')
    parser.add_argument('--bufferSize', type=int, default=10, help='Number of frames to buffer')
    parser.add_argument('--linearLayerIdx', type=int, default=0, help='Index of linear layer to load')
    parser.add_argument('--subsampleFactor', type=int, default=2, help='RNN 2nd layer subsample factor')
    parser.add_argument('--task', type=str, help='Task type', choices={"speech", "handwriting"})
    parser.add_argument('--blockMean', type=str, help='Block mean path')
    parser.add_argument('--blockStd', type=str, help='Block std path')

    args = parser.parse_args()

    print(f"starting Redis connection to {args.host}:{args.port}/{args.db}")
    r = redis.Redis(host=args.host, port=args.port, db=args.db)

    print(f"loading RNN from {args.rnnDir}")
    rnnModel, rnnArgs = singleStepRNN.loadRNN(args.rnnDir,
                                              checkpointIdx=args.checkpointIdx,
                                              loadLayerIdx=args.linearLayerIdx)

    blockMean = None
    blockStd = None
    if args.blockMean is not None:
        print(f"loading block mean from {args.blockMean}")
        blockMean = np.load(args.blockMean).astype(np.float32)
    if args.blockStd is not None:
        print(f"loading block std from {args.blockStd}")
        blockStd = np.load(args.blockStd).astype(np.float32)

    if args.lmDir is not None:
        print(f"loading LM from {args.lmDir}")
        ngramDecoder = lmDecoderUtils.build_lm_decoder(
            args.lmDir,
            acoustic_scale=1.6,
            nbest=1
        )

    print("running...")

    state = State.IDLE
    blankClass = 0
    blankTicks = 0
    buffer = np.zeros([1, args.bufferSize, args.inputDim], dtype=np.float32)
    rnnStates = [rnnModel['rnnLayers'].initStates, None]
    frameCounter = 0
    taskState = 0
    startT = time()
    while(1):
        # Get task state
        if r.get('task_state') is None:
            continue
        newTaskState = int(r.get('task_state').decode('utf-8'))

        # XREAD: Blocking request which waits on a stream for 'block' ms maximum
        stream = r.xread({'binned:neural:stream': '$'}, count=1, block=1000)
        if stream == []:
            continue
       
        #print(f'time elapsed {time() - startT}')
        startT = time()
        # Convert stream data to np, and put into buffer.
        bytesData = stream[0][1][0][1][b'data']
        npBinnedNeural = np.frombuffer(bytesData, dtype=np.uint8)
        npBinnedNeural = npBinnedNeural.astype(np.float32)
        # TODO: This should be done in simulator/matlab
        if args.task == 'handwriting':
            npBinnedNeural = npBinnedNeural[:192]
        assert npBinnedNeural.shape == (args.inputDim,)
        if blockMean is not None:
            npBinnedNeural = (npBinnedNeural - blockMean) / blockStd
        npBinnedNeural = npBinnedNeural[np.newaxis, np.newaxis, :]
        # Append current frame to the tail of the buffer and remove the head
        buffer = np.concatenate([buffer[:, 1:, :], npBinnedNeural], axis=1)

        # Task hasn't started yet. Wait for go cue.
        if taskState == 0 and newTaskState == 0:
            continue
        # Task has finished. If decoder is still running, need to force stop it.
        forceStop = False
        if taskState == 1 and newTaskState == 0:
            print('force stop')
            forceStop = True
        taskState = newTaskState

        # Wait until buffer is full
        if forceStop or (frameCounter >= args.bufferSize and frameCounter % args.subsampleFactor == 0):
            logits, rnnStates = singleStepRNN.runSingleStep(rnnModel,
                                                            tf.constant(buffer, dtype=tf.float32),
                                                            rnnStates)
            logits = logits.numpy()
            # LM decoder expects blank symbol at index 0
            if args.task == 'handwriting':
                logits = lmDecoderUtils.rearrange_handwriting_logits(logits)
            elif args.task == 'speech':
                logits = lmDecoderUtils.rearrange_speech_logits(logits)
            else:
                raise ValueError(f'Unknown task {args.task}')

            if np.argmax(logits, -1) == blankClass:
                blankTicks += 1
            else:
                blankTicks = 0

            if forceStop or (blankTicks > args.eosBlank and state == State.DECODING):
                state = State.IDLE
                buffer = np.zeros([1, args.bufferSize, args.inputDim], dtype=np.float32)
                rnnStates = [rnnModel['rnnLayers'].initStates, None]
                frameCounter = -1
                r.xadd('binned:decoderOutput:stream', {'end': ''})

                if args.lmDir is not None:
                    ngramDecoder.FinishDecoding()
                    if len(ngramDecoder.result()) > 0:
                        print(f'Final: {ngramDecoder.result()[0].sentence}')
                        if forceStop:
                            r.set('decoded_sentence', '')
                        else:
                            r.set('decoded_sentence', f'{ngramDecoder.result()[0].sentence}')
                        r.xadd('binned:decoderOutput:stream',
                              {'final_decoded_sentence': ngramDecoder.result()[0].sentence})
                    ngramDecoder.Reset()

            if blankTicks == 1 and state == State.IDLE:
                print('start decoding')
                state = State.DECODING
                r.xadd('binned:decoderOutput:stream', {'start': ''})

            if state == State.DECODING:
                r.xadd('binned:decoderOutput:stream', {'data': logits[0,0,:].tobytes(order='C') })

                if args.lmDir is not None:
                    lm_decoder.DecodeNumpy(ngramDecoder,logits[0]),
                    if len(ngramDecoder.result()) > 0:
                        print(f'Partial: {ngramDecoder.result()[0].sentence}')
                        r.set('decoded_sentence', f'{ngramDecoder.result()[0].sentence} ...')
                        r.xadd('binned:decoderOutput:stream',
                              {'partial_decoded_sentence': ngramDecoder.result()[0].sentence})

        #stop program when the block is over
        if r.exists('blockOver'):
            break

        frameCounter += 1

    r.close()

if __name__ == "__main__":
    main()
