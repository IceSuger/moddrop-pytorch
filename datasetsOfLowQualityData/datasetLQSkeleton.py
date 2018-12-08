from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal


class DatasetLQSkeleton(DatasetOfDamagedMultimodal):
    def __init__(self, root, train_valid_test='train'):
        super().__init__(root, train_valid_test=train_valid_test)

    def __getitem__(self, ind):
        data, label, QoU = self._my_getitem__(ind)
        # print(f'label = {label[0]}, type(label) = {type(label)}')
        return data['mocap'], label[0]