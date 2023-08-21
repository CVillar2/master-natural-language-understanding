import numpy as np

from char_embedding_model import CharEmbeddingModelBuilder


class ModelTrainer:

    config = None
    model = None

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def train(self, inputs, targets, validation_data=None, is_char=False):
        the_inputs = inputs
        if is_char:
            the_inputs = CharEmbeddingModelBuilder.prepare_inputs(self.config, inputs)
            if validation_data is not None:
                validation_data[0] = CharEmbeddingModelBuilder.prepare_inputs(self.config, validation_data[0])
        self.model.fit(the_inputs, targets, batch_size=self.config.batch_size,
                       epochs=self.config.max_epochs, validation_data=validation_data)
