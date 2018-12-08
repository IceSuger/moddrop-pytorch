import torch

from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal


class DatasetLQVideoFeatureExtractor(DatasetOfDamagedMultimodal):
    def __init__(self, root, train_valid_test='train'):
        super().__init__(root, train_valid_test=train_valid_test)

    def __getitem__(self, ind):
        data, label, QoU = self._my_getitem__(ind)
        retdata = []
        retdata.append(data['video'][0]) # [2018-12-7 v2.1] 这两行这样的取法，可能不对，暂时存疑
        retdata.append(data['video'][2]) #
        return torch.tensor(retdata), label[0]