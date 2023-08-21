
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras import backend as K


class ModelBuilder:

    config = None
    input_shape = None
    max_length = 0
    model = None
    num_classes = None
    verbose = False

    def __init__(self, config, input_shape, num_classes, verbose=False):
        self.config = config
        self.max_length = config.max_sentence_length
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.verbose = verbose

    def build(self, sentences):
        embedding_size = self.config.embedding_size
        text_vectorizer = keras.layers.TextVectorization(split=None, output_sequence_length=self.max_length,
                                                         standardize=None, pad_to_max_tokens=False, )
        text_vectorizer.adapt(sentences)
        if self.verbose:
            vectorized_text = text_vectorizer(sentences)
            print(vectorized_text)
            print(f'Text vectorizer vocabulary size = { text_vectorizer.vocabulary_size() }')
        inputs = Input(shape=self.input_shape, dtype=tf.string)
        v = text_vectorizer(inputs)
        e = Embedding(text_vectorizer.vocabulary_size(), embedding_size, mask_zero=True)(v)
        lstm = LSTM(self.max_length, return_sequences=True)(e)
        outputs = TimeDistributed(Dense(self.num_classes, activation=keras.activations.softmax))(lstm)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy', ignore_class_accuracy()])
        if self.verbose:
            model.summary()
        self.model = model
        return model


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.cast(y_true, 'int64')
        y_pred_class = K.argmax(y_pred, axis=2)
        ignore_mask = K.cast(K.not_equal(y_true_class, to_ignore), 'int64')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int64') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy
