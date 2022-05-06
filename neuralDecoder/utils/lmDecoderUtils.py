import os
from time import time

import numpy as np
from tqdm.notebook import trange, tqdm
from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel
import tensorflow as tf

import lm_decoder
import neuralDecoder.utils.rnnEval as rnnEval

'''
GPT2 Language Model Utils
'''
def build_gpt2(modelName='gpt2-xl', cacheDir=None):
    tokenizer = GPT2TokenizerFast.from_pretrained(modelName, cache_dir=cacheDir)
    model = TFGPT2LMHeadModel.from_pretrained(modelName, cache_dir=cacheDir)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def rescore_with_gpt2(model, tokenizer, hypotheses):
    inputs = tokenizer(hypotheses, return_tensors='tf', padding=True)
    outputs = model(inputs)
    logProbs = tf.math.log(tf.nn.softmax(outputs['logits'], -1))
    logProbs = logProbs.numpy()

    newLMScores = []
    B, T, _ = logProbs.shape
    for i in range(B):
        hyp = hypotheses[i]
        words = hyp.split(' ')

        newLMScore = 0.
        for j in range(1, len(words)):
            newLMScore += logProbs[i, j - 1, inputs['input_ids'][i, j].numpy()]

        newLMScores.append(newLMScore)

    return newLMScores

def gpt2_lm_decode(model, tokenizer, nbestOutputs, acousticScale):
    decodedSentences = []
    for i in trange(len(nbestOutputs)):
        hypotheses = []
        acousticScores = []
        for out in nbestOutputs[i]:
            hyp = out[0].strip()
            if len(hyp) == 0:
                continue
            hyp = hyp.replace('>', '')
            hyp = hyp.replace('  ', ' ')
            hyp = hyp.replace(' ,', ',')
            hyp = hyp.replace(' .', '.')
            hyp = hyp.replace(' ?', '?')
            hypotheses.append(hyp)
            acousticScores.append(out[1])

        if len(hypotheses) == 0:
            decodedSentences.append("")
            continue

        acousticScores = np.array(acousticScores)
        newLMScores = np.array(rescore_with_gpt2(model, tokenizer, hypotheses))

        maxIdx = np.argmax(newLMScores + acousticScale * acousticScores)
        decodedSentences.append(hypotheses[maxIdx])

    return decodedSentences

def cer_with_gpt2_decoder(model, tokenizer, nbestOutputs, acousticScale, inferenceOut):

    decodedSentences = gpt2_lm_decode(model, tokenizer, nbestOutputs, acousticScale)
    trueSentences = _extract_true_sentences(inferenceOut)
    trueSentencesProcessed = []
    for trueSent in trueSentences:
        trueSent = trueSent.replace('>',' ')
        trueSent = trueSent.replace('~','.')
        trueSent = trueSent.replace('#','')
        trueSentencesProcessed.append(trueSent)

    cer, wer = _cer_and_wer(decodedSentences, trueSentencesProcessed)

    return {
        'cer': cer,
        'wer': wer,
        'decoded_transcripts': decodedSentences
    }


'''
NGram Language Model Utils
'''
def build_lm_decoder(model_path,
                     max_active=7000,
                     min_active=200,
                     beam=17.,
                     lattice_beam=8.,
                     acoustic_scale=1.5,
                     ctc_blank_skip_threshold=1.0,
                     nbest=1):
    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        nbest
    )

    decode_resource = lm_decoder.DecodeResource(
        os.path.join(model_path, 'TLG.fst'),
        "",
        "",
        os.path.join(model_path, 'words.txt'),
        ""
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

    return decoder

def lm_decode(decoder, logits, returnNBest=False):
    assert len(logits.shape) == 2

    lm_decoder.DecodeNumpy(decoder, logits)
    decoder.FinishDecoding()

    if not returnNBest:
        decoded = decoder.result()[0].sentence
    else:
        decoded = []
        for r in decoder.result():
            decoded.append((r.sentence, r.ac_score, r.lm_score))

    decoder.Reset()

    return decoded

def cer_with_lm_decoder(decoder, inferenceOut, includeSpaceSymbol=True,
                        outputType='handwriting'):
    # Reshape logits to kaldi order
    logits = inferenceOut['logits']
    if outputType == 'handwriting':
        logits = rearrange_handwriting_logits(logits, includeSpaceSymbol)
        trueSentences = _extract_true_sentences(inferenceOut)
    elif outputType == 'speech':
        logits = rearrange_speech_logits(logits)
        trueSentences = _extract_transcriptions(inferenceOut)

    # Decode with language model
    decodedSentences = []
    decodeTime = []
    for l in trange(len(inferenceOut['logits'])):
        decoder.Reset()

        logitLen = inferenceOut['logitLengths'][l]
        start = time()
        decoded = lm_decode(decoder, logits[l, :logitLen])

        # Post-process
        if outputType == 'handwriting':
            if includeSpaceSymbol:
                decoded = decoded.replace(' ', '')
            else:
                decoded = decoded.replace(' ', '>')
            if len(decoded) > 0 and decoded[-1] == '.':
                decoded = decoded[:-1] + '~'
        elif outputType == 'speech':
            decoded = decoded.strip()

        decodeTime.append((time() - start) * 1000)
        decodedSentences.append(decoded)

    assert len(trueSentences) == len(decodedSentences)

    cer, wer = _cer_and_wer(decodedSentences, trueSentences, outputType)

    return {
        'cer': cer,
        'wer': wer,
        'decoded_transcripts': decodedSentences,
        'decode_time': decodeTime
    }

def rearrange_handwriting_logits(logits, includeSpaceSymbol=True):
    char_range = list(range(0, 26))
    if includeSpaceSymbol:
        symbol_range = [26, 27, 30, 29, 28]
    else:
        symbol_range = [27, 30, 29, 28]
    logits = logits[:, :, [31] + symbol_range + char_range]
    return logits

def rearrange_speech_logits(logits):
    logits = np.concatenate([logits[:, :, -1:], logits[:, :, :-1]], axis=-1)
    return logits

def _cer_and_wer(decodedSentences, trueSentences, outputType='handwriting'):
    allCharErr = 0
    allChar = 0
    allWordErr = 0
    allWord = 0
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = rnnEval.wer([c for c in trueSent], [c for c in decSent])
        if outputType == 'handwriting':
            trueWords = trueSent.replace(">", " > ").split(" ")
            decWords = decSent.replace(">", " > ").split(" ")
        elif outputType == 'speech':
            trueWords = trueSent.split(" ")
            decWords = decSent.split(" ")
        nWordErr = rnnEval.wer(trueWords, decWords)

        allCharErr += nCharErr
        allWordErr += nWordErr
        allChar += len(trueSent)
        allWord += len(trueWords)

    cer = allCharErr / float(allChar)
    wer = allWordErr / float(allWord)

    return cer, wer

def _extract_transcriptions(inferenceOut):
    transcriptions = []
    for i in range(len(inferenceOut['transcriptions'])):
        endIdx = np.argwhere(inferenceOut['transcriptions'][i] == 0)[0, 0]
        trans = ''
        for c in range(endIdx):
            trans += chr(inferenceOut['transcriptions'][i][c])
        transcriptions.append(trans)

    return transcriptions

def _extract_true_sentences(inferenceOut):
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 '>',',',"'",'~','?']

    trueSentences = []
    for i in range(len(inferenceOut['trueSeqs'])):
        trueSent = ''
        endIdx = np.argwhere(inferenceOut['trueSeqs'][i] == -1)
        endIdx = endIdx[0,0]
        for c in range(endIdx):
            trueSent += charMarks[inferenceOut['trueSeqs'][i][c]]
        trueSentences.append(trueSent)

    return trueSentences