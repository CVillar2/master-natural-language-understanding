
import dataclasses


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False


@dataclasses.dataclass
class POSTaggerConfig:

    batch_size: int = 64
    embedding_size: int = 64
    num_sentences_to_predict: int = 2
    max_sentence_length: int = 128
    max_word_length: int = 20
    max_sentences: int = 100000 if IN_COLAB else 256
    max_epochs: int = 20 if IN_COLAB else 2
