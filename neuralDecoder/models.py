import tensorflow as tf
from tensorflow.keras import Model


class GRU(Model):
    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0):
        super(GRU, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        #actReg = tf.keras.regularizers.L2(actReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional

        if bidirectional:
            self.initStates = [
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
            ]
        else:
            self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)))

        self.rnn1 = tf.keras.layers.GRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        kernel_regularizer=weightReg,
                                        activity_regularizer=actReg,
                                        recurrent_initializer=recurrent_init,
                                        kernel_initializer=kernel_init,
                                        dropout=dropout)
        self.rnn2 = tf.keras.layers.GRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        kernel_regularizer=weightReg,
                                        activity_regularizer=actReg,
                                        recurrent_initializer=recurrent_init,
                                        kernel_initializer=kernel_init,
                                        dropout=dropout)
        if bidirectional:
            self.rnn1 = tf.keras.layers.Bidirectional(self.rnn1)
            self.rnn2 = tf.keras.layers.Bidirectional(self.rnn2)
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, state=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        if state is None:
            if self.bidirectional:
                initState1 = [tf.tile(s, [batchSize, 1]) for s in self.initStates]
            else:
                initState1 = tf.tile(self.initStates, [batchSize, 1])
            initState2 = None
        else:
            initState1 = state[0]
            initState2 = state[1]

        x, s1 = self.rnn1(x, training=training, initial_state=initState1)
        if self.subsampleFactor > 1:
            x = x[:, ::self.subsampleFactor, :]
        x, s2 = self.rnn2(x, training=training, initial_state=initState2)
        x = self.dense(x, training=training)

        if returnState:
            return x, [s1, s2]
        else:
            return x

    def getIntermediateLayerOutput(self, x):
        x, _ = self.rnn1(x)
        return x

