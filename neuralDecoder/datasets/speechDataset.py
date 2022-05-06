import pathlib
import random

import tensorflow as tf

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

CONSONANT_DEF = ['CH', 'SH', 'JH', 'R', 'B',
                 'M',  'W',  'V',  'F', 'P',
                 'D',  'N',  'L',  'S', 'T',
                 'Z',  'TH', 'G',  'Y', 'HH',
                 'K', 'NG', 'ZH', 'DH']
VOWEL_DEF = ['EY', 'AE', 'AY', 'EH', 'AA',
             'AW', 'IY', 'IH', 'OY', 'OW',
             'AO', 'UH', 'AH', 'UW', 'ER']

class SpeechDataset():
    def __init__(self,
                 rawFileDir,
                 nInputFeatures,
                 nClasses,
                 maxSeqElements,
                 bufferSize,
                 syntheticFileDir=None,
                 syntheticMixingRate=0.33,
                 subsetSize=-1,
                 labelDir=None
                 ):
        self.rawFileDir = rawFileDir
        self.nInputFeatures = nInputFeatures
        self.nClasses = nClasses
        self.maxSeqElements = maxSeqElements
        self.bufferSize = bufferSize
        self.syntheticFileDir = syntheticFileDir
        self.syntheticMixingRate = syntheticMixingRate

    def build(self, batchSize, isTraining):
        def _loadDataset(fileDir):
            files = [str(x) for x in pathlib.Path(fileDir).glob("*.tfrecord")]
            if isTraining:
                random.shuffle(files)

            dataset = tf.data.TFRecordDataset(files)
            return dataset

        print(f'Load data from {self.rawFileDir}')
        rawDataset = _loadDataset(self.rawFileDir)
        if self.syntheticFileDir and self.syntheticMixingRate > 0:
            print(f'Load data from {self.syntheticFileDir}')
            syntheticDataset = _loadDataset(self.syntheticFileDir)
            dataset = tf.data.experimental.sample_from_datasets(
                [rawDataset.repeat(), syntheticDataset.repeat()],
                weights=[1.0 - self.syntheticMixingRate, self.syntheticMixingRate])
        else:
            dataset = rawDataset

        datasetFeatures = {
            "inputFeatures": tf.io.FixedLenSequenceFeature([self.nInputFeatures], tf.float32, allow_missing=True),
            #"classLabelsOneHot": tf.io.FixedLenSequenceFeature([self.nClasses+1], tf.float32, allow_missing=True),
            "newClassSignal": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "ceMask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "seqClassIDs": tf.io.FixedLenFeature((self.maxSeqElements), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "transcription": tf.io.FixedLenFeature((self.maxSeqElements), tf.int64)
        }

        def parseDatasetFunction(exampleProto):
            return tf.io.parse_single_example(exampleProto, datasetFeatures)

        dataset = dataset.map(parseDatasetFunction, num_parallel_calls=tf.data.AUTOTUNE)

        if isTraining:
            # Use all elements to adapt normalization layer
            datasetForAdapt = dataset.map(lambda x: x['inputFeatures'] + 0.001,
                num_parallel_calls=tf.data.AUTOTUNE)

            # Shuffle and transform data if training
            dataset = dataset.shuffle(self.bufferSize)
            if self.syntheticMixingRate == 0:
                dataset = dataset.repeat()
            dataset = dataset.padded_batch(batchSize)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset, datasetForAdapt
        else:
            dataset = dataset.padded_batch(batchSize)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
