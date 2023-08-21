
import numpy as np


class ModelPredictor:

    model = None
    classes = None
    tag_decoder = None
    verbose = False

    def __init__(self, model, classes, tag_decoder, verbose=False):
        self.model = model
        self.classes = classes
        self.tag_decoder = tag_decoder
        self.verbose = verbose

    def predict(self, inputs, unpadded_inputs):
        predicted_tags = self.model.predict(inputs)
        predicted_tags = np.argmax(predicted_tags, axis=2)
        predicted_tags_decoded = np.asarray([self.tag_decoder(pt) for pt in predicted_tags])
        predicted_tags_list = [predicted_tags_decoded[i, 0:len(unpadded_inputs[i])] for i in range(len(predicted_tags))]
        if self.verbose:
            if isinstance(inputs, dict):
                print("Inputs shape WORDS = ", inputs["words"].shape)
                print("Inputs shape CHARS = ", inputs["chars"].shape)
            else:
                print("Inputs shape = ", inputs.shape)
            print("Predicted shape = ", predicted_tags_decoded.shape)
            print(predicted_tags_list)
        return predicted_tags_list, predicted_tags_decoded
