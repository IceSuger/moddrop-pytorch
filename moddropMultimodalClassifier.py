from multimodalClassifier import multimodalClassifier


class moddropMultimodalClassifier(multimodalClassifier):
    def __init__(self, input_folder, filter_folder, number_of_classes=21,
                 step=4, nframes=5, block_size=36, batch_size=42, pretrained=False):
        multimodalClassifier.__init__(self, input_folder, filter_folder, number_of_classes=number_of_classes,
                 step=step, nframes=nframes, block_size=block_size, batch_size=batch_size, pretrained=pretrained)

        # 加载在未经 moddrop 的数据集上已经完成训练的 multimodalClassifier 参数
        name = 'checkpoints/' + 'multimodalClassifier' + '.pth'
        self.load_weights(name)

        self.model_name = 'moddropMultimodalClassifier'

        # [Xiao] 指定与训练的权重文件位置
        pretrainedFolder = 'checkpoints/'
        self.pretrainedPaths['moddropMultimodal'] = pretrainedFolder + 'moddropMultimodalClassifier.pth'