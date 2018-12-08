from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal


class DatasetLQAudio(DatasetOfDamagedMultimodal):
    def __init__(self, root, train_valid_test='train'):
        super().__init__(root, train_valid_test=train_valid_test)

    def __getitem__(self, ind):
        data, label, QoU = self._my_getitem__(ind)
        return data['audio'], label[0]