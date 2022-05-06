import re
import scipy.io
import numpy as np

def loadLexicon(lexiconFile, emaFile):
    #lexiconFile = '/oak/stanford/groups/shenoy/fwillett/speech/cmudict-0.7b.txt'
    lexDict = {}
    for line in open(lexiconFile):
        if not line.startswith(';;;'):
            lineParts = line.split('  ')
            key = lineParts[0].lower()
            phonemes = lineParts[1].lower()
            phonemes = re.sub("[^a-z .',?]", '', phonemes)
            phonemes = phonemes.split(' ')
            lexDict[key] = phonemes
            
            
    ema = scipy.io.loadmat('/oak/stanford/groups/shenoy/fwillett/speech/emaPhonemeAverages.mat')
    pIDDict = {}
    for x in range(ema['vectorMeans_norm'].shape[0]):
        pName = str(ema['phonemeList'][x,0][0])
        pIDDict[pName] = x+1
    
    return lexDict, pIDDict

def charToPhoneme(sentence, lexDict, pIDDict):
    words = sentence.split(' ')
    phonemes = []
    phonemeIDs = []
    for w in range(len(words)):
        thesePhonemes = lexDict[words[w]]
        
        phonemes.extend(thesePhonemes)
        for x in range(len(thesePhonemes)):
            phonemeIDs.append(pIDDict[thesePhonemes[x]])
        
    return phonemes, np.array(phonemeIDs)

