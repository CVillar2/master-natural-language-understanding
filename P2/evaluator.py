
class ModelEvaluator:

    model = None
    classes = None
    verbose = False

    def __init__(self, model, classes, verbose=False):
        self.model = model
        self.classes = classes
        self.verbose = verbose

    def evaluate(self, inputs, targets):
        results = self.model.evaluate(inputs, targets, batch_size=64)
        if self.verbose:
            print("Loss = ", results[0])
            print("Accuracy = ", results[1])
            print("Accuracy without blanks = ", results[2])
        return results
