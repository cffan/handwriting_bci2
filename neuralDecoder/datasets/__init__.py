from .handwritingDataset import HandwritingDataset
from .speechDataset import SpeechDataset

def getDataset(datasetName):
    if datasetName == 'handwriting':
        return HandwritingDataset
    elif datasetName == 'speech':
        return SpeechDataset
    else:
        raise ValueError('Dataset not found')