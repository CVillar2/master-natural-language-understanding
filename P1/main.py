import re

import keras.preprocessing.text
import numpy as np

from argparser import ArgumentParser
from char_embedding_model import CharEmbeddingModelBuilder
from config import POSTaggerConfig
from dataset import DatasetParser, DatasetFiles
from evaluator import ModelEvaluator
from base_model import ModelBuilder
from predictor import ModelPredictor
from trainer import ModelTrainer


def main():
    print("NLU Assignment 1")
    # Configure global parameters
    config = POSTaggerConfig()
    num_sentences = config.max_sentences
    # Parse command-line arguments
    args = ArgumentParser()
    args.parse()
    # Parse datasets
    all_classes, num_classes, test_file, train_set, validation_file = parse_datasets(args, config)
    # Build the model
    model = build_model(args, config, num_classes, train_set)
    # Training model with training set
    validation_set = DatasetParser(config, validation_file).parse() if args.validate else None
    train(args, config, model, num_sentences, train_set, validation_set)
    # Compute validation metrics for the model
    validate(args, config, all_classes, model, num_sentences, validation_file)
    # Predict tags for given inputs
    predict(args, config, all_classes, model, train_set)
    # Evaluate model with test set
    evaluate(args, config, all_classes, model, num_sentences, test_file)
    # Tag a given sentence
    tag(args, config, all_classes, model, train_set)


def parse_datasets(args, config):
    train_file = None
    validation_file = None
    test_file = None
    if args.dataset is not None:
        dataset = DatasetFiles[args.dataset].value
        train_file, validation_file, test_file = dataset.load()
    if args.train_file is not None:
        train_file = args.train_file
    if args.validation_file is not None:
        validation_file = args.validation_file
    if args.test_file is not None:
        test_file = args.test_file
    train_set = DatasetParser(config, train_file, args.verbose).parse()
    all_classes = train_set.classes
    num_classes = train_set.num_classes
    return all_classes, num_classes, test_file, train_set, validation_file


def build_model(args, config, num_classes, train_set):
    if args.model == "char":
        builder = CharEmbeddingModelBuilder(config, (train_set.padded_sentences.shape[1],), num_classes, args.verbose)
        model = builder.build(train_set.padded_sentences)
    else:
        builder = ModelBuilder(config, (train_set.padded_sentences.shape[1],), num_classes, args.verbose)
        model = builder.build(train_set.padded_sentences)
    return model


def train(args, config, model, num_sentences, train_set, validation_set):
    if not args.train:
        return
    if args.verbose:
        print("Training...")
        print(f'Training set shape = {train_set.padded_sentences.shape}')
        print(f'Encoded tags shape = {train_set.padded_encoded_tags.shape}')
    trainer = ModelTrainer(config, model)
    train_inputs = train_set.padded_sentences[0:num_sentences]
    train_targets = train_set.padded_encoded_tags[0:num_sentences]
    validation_data = None
    if args.validate:
        validation_inputs = validation_set.padded_sentences[0:num_sentences]
        validation_targets = validation_set.padded_encoded_tags[0:num_sentences]
        validation_data = (validation_inputs, validation_targets)
    trainer.train(train_inputs, train_targets, validation_data=validation_data, is_char=(args.model == "char"))


def validate(args, config, all_classes, model, num_sentences, validation_file):
    if not args.validate:
        return
    if args.verbose:
        print("Validating...")
    validation_set = DatasetParser(config, validation_file).parse()
    evaluator = ModelEvaluator(model, all_classes)
    validation_inputs = validation_set.padded_sentences[0:num_sentences]
    validation_targets = validation_set.padded_encoded_tags[0:num_sentences]
    if args.model == "char":
        validation_inputs = CharEmbeddingModelBuilder.prepare_inputs(config, validation_inputs)
    evaluator.evaluate(validation_inputs, validation_targets)


def predict(args, config, all_classes, model, train_set):
    if not args.predict:
        return
    if args.verbose:
        print("Predicting from training set...")
    inputs_to_predict = train_set.sentences[0:config.num_sentences_to_predict]
    inputs_to_predict_padded = train_set.padded_sentences[0:config.num_sentences_to_predict]
    predictor = ModelPredictor(model, all_classes, train_set.tag_decoder, args.verbose)
    if args.model == "char":
        inputs_to_predict_padded = CharEmbeddingModelBuilder.prepare_inputs(config, inputs_to_predict_padded)
    predicted_outputs, predicted_outputs_padded = predictor.predict(inputs_to_predict_padded, inputs_to_predict)
    if args.verbose:
        for i in range(len(inputs_to_predict)):
            input_i = inputs_to_predict[i]
            output_i = predicted_outputs[i]
            print(" ".join(input_i))
            for j in range(len(input_i)):
                print(f'\t{input_i[j]} => {output_i[j]} (real = {train_set.tags[i][j]})')


def tag(args, config, all_classes, model, train_set):
    if not args.tag or not args.sentence:
        return
    sentence = args.sentence
    sentence_words = re.findall(r"[\w']+|[.,!?;]", sentence)
    if args.verbose:
        print(f'Tagging: "{ args.sentence }" as { sentence_words }')
    inputs_to_predict = np.asarray([sentence_words])
    input_length = len(sentence_words)
    input_to_predict_padded = sentence_words if input_length == config.max_sentence_length else \
        sentence_words + ["<BLANK>" for _ in range(config.max_sentence_length-input_length)]
    inputs_to_predict_padded = np.asarray([input_to_predict_padded])
    predictor = ModelPredictor(model, all_classes, train_set.tag_decoder, args.verbose)
    if args.model == "char":
        inputs_to_predict_padded = CharEmbeddingModelBuilder.prepare_inputs(config, inputs_to_predict_padded)
    predicted_outputs, predicted_outputs_padded = predictor.predict(inputs_to_predict_padded, inputs_to_predict)
    input_i = inputs_to_predict[0]
    output_i = predicted_outputs[0]
    print(" ".join(input_i))
    for j in range(len(input_i)):
        print(f'\t{input_i[j]} => {output_i[j]}')


def evaluate(args, config, all_classes, model, num_sentences, test_file):
    if not args.evaluate:
        return
    if args.verbose:
        print("Evaluating...")
    test_set = DatasetParser(config, test_file).parse()
    evaluator = ModelEvaluator(model, all_classes, args.verbose)
    eval_inputs = test_set.padded_sentences[0:num_sentences]
    eval_targets = test_set.padded_encoded_tags[0:num_sentences]
    if args.model == "char":
        eval_inputs = CharEmbeddingModelBuilder.prepare_inputs(config, eval_inputs)
    evaluator.evaluate(eval_inputs, eval_targets)


if __name__ == "__main__":
    main()
