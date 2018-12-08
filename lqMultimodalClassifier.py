from multimodalClassifier import multimodalClassifier


class lqMultimodalClassifier(multimodalClassifier):
    def __init__(self, input_folder, filter_folder, number_of_classes=21,
                 step=4, nframes=5, block_size=36, batch_size=42, pretrained=False):
        super().__init__(input_folder, filter_folder)
        self.model_name = 'lqMultimodalClassifier'
