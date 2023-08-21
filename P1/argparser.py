
import argparse


class ArgumentParser:

    train = False
    validate = False
    evaluate = False
    predict = False
    tag = False
    sentence = ""
    dataset = None
    model = None
    train_file = None
    validation_file = None
    test_file = None
    verbose = False
    parser = None

    def __init__(self):
        parser = argparse.ArgumentParser(description='NLU A1')
        parser.add_argument('-train', action="store_true", help="Train the selected model")
        parser.add_argument('-validate', action="store_true", help="Validate the selected model with the validation set")
        parser.add_argument('-eval', action="store_true", help="Evaluate the selected model with the test set")
        parser.add_argument('-predict', action="store_true", help="Use the model to predict tags from one of the sets")
        parser.add_argument('-verbose', action="store_true", help="Verbose mode to output extra execution info")
        parser.add_argument('-model', action="store", default="base",
                            help="The model to use: either 'base' or 'char' to use the char embedding")
        parser.add_argument('-tag', action="store_true",
                            help="Predict the POS tags for a given sentence. To use with the \"-sentence\" parameter")
        parser.add_argument('-dataset_file', action="store",
                            help="The path to the CoNLL-U dataset files without the "
                                 "\"-dev.conllu\" extension. Example: \"UD_English_EWT\\en_ewt-ud\"")
        parser.add_argument('-sentence', action="store",
                            help="The sentence to be POS-tagged when \"-tag\" is passed.")
        parser.add_argument('-dataset', action="store",
                            help="The preconfigured dataset to use",
                            choices=['UD_English_EWT', 'UD_Spanish_AnCora', 'UD_French_GSD', 'UD_Galician_CTG'],
                            default="UD_English_EWT",)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        self.train = args.train
        self.validate = args.validate
        self.evaluate = args.eval
        self.predict = args.predict
        self.tag = args.tag
        self.sentence = args.sentence
        self.verbose = args.verbose
        self.model = args.model
        self.dataset = args.dataset
        dataset_file = args.dataset_file
        if dataset_file is not None:
            self.train_file = f'{dataset_file}-train.conllu'
            self.validation_file = f'{dataset_file}-dev.conllu'
            self.test_file = f'{dataset_file}-test.conllu'
        if args.verbose:
            print(args)
