import dataclasses
from enum import Enum
from typing import Any

import keras.utils
import numpy as np
import tensorflow as tf
from conllu import parse
from sklearn.preprocessing import LabelEncoder


class DatasetParser:
    config = None
    dataset = None
    file = ""
    verbose = False

    def __init__(self, config, file, verbose=False):
        self.config = config
        self.file = file
        self.verbose = verbose

    def parse(self):
        max_length = self.config.max_sentence_length
        data_file = open(self.file, "r", encoding="utf-8")
        data = data_file.read()
        parsed = parse(data, fields=["id", "form", "lemma", "tags"])
        sentences = list()
        padded_sentences = list()
        tags = list()
        padded_tags = list()
        for sentence in parsed:
            sentence_forms = list([item["form"] for item in sentence])
            padded_sentence_forms = sentence_forms.copy()
            sentence_tags = list([item["tags"] for item in sentence])
            padded_sentence_tags = sentence_tags.copy()
            if len(sentence_forms) > max_length:
                continue
            if len(sentence_forms) < max_length:
                pad = max_length - len(sentence_forms)
                for i in range(pad):
                    padded_sentence_forms.append("")
                    padded_sentence_tags.append("<BLANK>")
            sentences.append(sentence_forms)
            padded_sentences.append(padded_sentence_forms)
            tags.append(sentence_tags)
            padded_tags.append(padded_sentence_tags)
        sentences = sentences
        padded_sentences = np.array(padded_sentences)
        tags = tags
        padded_tags = np.array(padded_tags)
        encoded_tags, padded_encoded_tags, label_encoder = prepare_targets(padded_tags, max_length, sentences)
        classes = label_encoder.classes_
        num_classes = len(label_encoder.classes_)

        def tag_decoder(enc_tags):
            return label_encoder.inverse_transform(enc_tags)

        dataset = Dataset(sentences, padded_sentences, tags, padded_tags, encoded_tags,
                          padded_encoded_tags, classes, num_classes, tag_decoder)
        self.dataset = dataset
        return dataset


@dataclasses.dataclass
class Dataset:
    sentences: Any
    padded_sentences: Any
    tags: Any
    padded_tags: Any
    encoded_tags: Any
    padded_encoded_tags: Any
    classes: Any
    num_classes: Any
    tag_decoder: Any


def prepare_targets(y_train, max_length, inputs, verbose=False):
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen=max_length, padding="post",
                                                             dtype=object, value="<BLANK>")
    if verbose:
        print(y_padded)
    label_encoder = LabelEncoder()
    yt = y_padded.reshape(-1, y_train.shape[1]).ravel()
    label_encoder.fit(yt)
    y_train_enc = label_encoder.transform(yt)
    padded_targets = y_train_enc.reshape(-1, y_train.shape[1])
    unpadded_targets = []
    for i in range(y_train.shape[0]):
        input_i = inputs[i]
        unpadded_target = []
        for j in range(len(input_i)):
            unpadded_target.append(padded_targets[i, j])
        unpadded_targets.append(unpadded_target)
    return unpadded_targets, padded_targets, label_encoder


@dataclasses.dataclass
class DatasetFile:
    name: str
    url_train: str
    url_validation: str
    url_test: str

    def load(self):
        path_train = keras.utils.get_file(origin=self.url_train)
        path_validation = keras.utils.get_file(origin=self.url_validation)
        path_test = keras.utils.get_file(origin=self.url_test)
        return path_train, path_validation, path_test


class DatasetFiles(Enum):
    UD_English_EWT = DatasetFile(
        name="UD_English_EWT",
        url_train="https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-train.conllu",
        url_validation="https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-dev.conllu",
        url_test="https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-test.conllu"
    )
    UD_Spanish_AnCora = DatasetFile(
        name="UD_Spanish_AnCora",
        url_train="https://github.com/UniversalDependencies/UD_Spanish-AnCora/raw/master/es_ancora-ud-train.conllu",
        url_validation="https://github.com/UniversalDependencies/UD_Spanish-AnCora/raw/master/es_ancora-ud-dev.conllu",
        url_test="https://github.com/UniversalDependencies/UD_Spanish-AnCora/raw/master/es_ancora-ud-test.conllu"
    )
    UD_French_GSD = DatasetFile(
        name="UD_French_GSD",
        url_train="https://github.com/UniversalDependencies/UD_French-GSD/raw/master/fr_gsd-ud-train.conllu",
        url_validation="https://github.com/UniversalDependencies/UD_French-GSD/raw/master/fr_gsd-ud-dev.conllu",
        url_test="https://github.com/UniversalDependencies/UD_French-GSD/raw/master/fr_gsd-ud-test.conllu"
    )
    UD_Galician_CTG = DatasetFile(
        name="UD_Galician_CTG",
        url_train="https://github.com/UniversalDependencies/UD_Galician-CTG/raw/master/gl_ctg-ud-train.conllu",
        url_validation="https://github.com/UniversalDependencies/UD_Galician-CTG/raw/master/gl_ctg-ud-dev.conllu",
        url_test="https://github.com/UniversalDependencies/UD_Galician-CTG/raw/master/gl_ctg-ud-test.conllu"
    )
