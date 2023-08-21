import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Bidirectional
from keras import backend as K
from keras import layers
import numpy as np

class CharEmbeddingModelBuilder:

    config = None
    input_shape = None
    max_sentence_length = 0
    model = None
    num_classes = None
    verbose = False
    max_word_length = 20

    def __init__(self, config, input_shape, num_classes, verbose=False):
        self.config = config
        self.max_sentence_length = config.max_sentence_length
        self.max_word_length = config.max_word_length
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.verbose = verbose
        

    def build(self, sentences):
        embedding_size = self.config.embedding_size
        word_vectorizer = keras.layers.TextVectorization(split=None, output_sequence_length=self.max_sentence_length,
                                                         standardize=None, pad_to_max_tokens=False, )
        word_vectorizer.adapt(sentences)
        if self.verbose:
            vectorized_word = word_vectorizer(sentences)
            print(vectorized_word)
            print(f'Text vectorizer vocabulary size = { word_vectorizer.vocabulary_size() }')
        words, len_char_set = self.words_to_char_array(self.config, sentences)
        words_input = Input(shape=self.input_shape, dtype=tf.string, name="words")
        chars_input = Input(shape=(self.max_sentence_length, self.max_word_length), dtype=tf.float64, name="chars")
        vectorized_words = word_vectorizer(words_input)
        word_embedding = Embedding(word_vectorizer.vocabulary_size(), embedding_size, mask_zero=True)(vectorized_words)
        character_embedding = TimeDistributed(Embedding(len_char_set, embedding_size, mask_zero=True))(chars_input)
        character_lstm = TimeDistributed(LSTM(self.max_word_length, return_sequences=False))(character_embedding)
        concatenated = layers.concatenate([word_embedding, character_lstm])
        lstm = Bidirectional(LSTM(self.max_sentence_length, return_sequences=True))(concatenated)
        outputs = TimeDistributed(Dense(self.num_classes, activation=keras.activations.softmax))(lstm)
        model_inputs = [words_input, chars_input]
        model = keras.Model(model_inputs, outputs)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy', ignore_class_accuracy()])
        if self.verbose:
            model.summary()
            #try:
            #keras.utils.plot_model(model, "model.png", show_shapes=True)
            #except:
                #print("Could not generate model graph!")
        self.model = model
        return model

    @staticmethod
    def prepare_inputs(config, inputs):
        char_input, _ = CharEmbeddingModelBuilder.words_to_char_array(config, inputs)
        the_inputs = {"words": inputs, "chars": char_input}
        return the_inputs

    @staticmethod
    def words_to_char_array(config, sentences):
        max_length = config.max_sentence_length
        max_word_length = config.max_word_length
        size = sentences.shape
        num_sentences = size[0]
        words = np.zeros((num_sentences, max_length, max_word_length), dtype=float)
        charset = set()
        for i in range(num_sentences):
            for j in range(max_length):
                word_ij = sentences[i, j]
                for k in range(max_word_length):
                    char_ijk = ord(word_ij[k]) if k < len(word_ij) else 0
                    charset.add(char_ijk)
        sorted_charset = sorted(charset)
        char_map = {}
        for i in range(len(sorted_charset)):
            char_map[sorted_charset[i]] = i
        for i in range(num_sentences):
            for j in range(max_length):
                word_ij = sentences[i, j]
                for k in range(max_word_length):
                    char_ijk = ord(word_ij[k]) if k < len(word_ij) else 0
                    words[i, j, k] = char_map[char_ijk]
        len_char_set = len(charset)
        return words, len_char_set


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.cast(y_true, 'int64')
        y_pred_class = K.argmax(y_pred, axis=2)
        ignore_mask = K.cast(K.not_equal(y_true_class, to_ignore), 'int64')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int64') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

