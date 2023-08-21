
# Natural Language Understanding - Assignment 1 - POS-Tagger
## Master in Artificial Intelligence - 2022/2023

Students:
- Carlos Villar Martínez
- Ovidio Manteiga Moar


# User manual

## Setup

The Python version used is Python `3.9.13`.

To set up the required dependencies, run the following command:

`pip install -r requirements.txt`

## Run the program

To run the program use the following command:

`python main.py [arguments]`

See usage section below for more information on the available command line parameters.

## Command line interface usage 

Run the program with the `-h` option to see the help:

`python main.py -h`

```
usage: main.py [-h] [-train] [-validate] [-eval] [-predict] [-verbose] [-model MODEL] [-tag] [-dataset_file DATASET_FILE] [-sentence SENTENCE]
               [-dataset {UD_English_EWT,UD_Spanish_AnCora,UD_French_GSD,UD_Galician_CTG}]
```

Optional arguments:

1. `-h, --help`: Show this help message and exit.
1. `-train`: Train the selected model.
1. `-validate`: Validate the selected model with the validation set.
1. `-eval`: Evaluate the selected model with the test set.
1. `-predict`: Use the model to predict tags from one of the sets.
1. `-verbose`: Verbose mode to output extra execution info.
1. `-model MODEL`: The model to use: either `base` or `char` to use the char embedding.
1. `-tag`: Predict the POS tags for a given sentence. To use with the `-sentence` parameter.
1. `-dataset_file DATASET_FILE`: The path to the local CoNLL-U dataset files without the `-dev.conllu` extension. Example: `UD_English_EWT\en_ewt-ud`.
1. `-sentence SENTENCE`: The sentence to be POS-tagged when "-tag" is passed.
1. `-dataset`: The preconfigured dataset to use, which is automatically downloaded. One of:
   1. `UD_English_EWT`
   1. `UD_Spanish_AnCora`
   1. `UD_French_GSD`
   1. `UD_Galician_CTG`

# Run in Google Colab

1. Create a new notebook in Google Colab or import the provided one `NLU_P1.ipynb`.
1. Connect to the runtime.
1. Copy all the Python files (`*.py`) to the runtime base folder.
1. Run the notebook.

For example, a code cell the following 2 lines to start with:

```commandline
!pip install -r requirements.txt
!python main.py -h
```

