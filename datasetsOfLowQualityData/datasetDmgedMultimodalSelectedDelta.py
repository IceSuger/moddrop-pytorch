from datasetsOfLowQualityData.datasetSelectedMultimodal import DatasetSelectedMultimodal


class DatasetDmgedMultimodalSelectedDelta(DatasetSelectedMultimodal):
    def __init__(self, root, phi_s, train_valid_test='train', QoU2delta_df=None):
        DatasetSelectedMultimodal.__init__(self, root, phi_s, train_valid_test=train_valid_test, QoU2delta_df=QoU2delta_df)

    def __getitem__(self, ind):
        data, label, QoU = self._my_getitem__(ind)
        selectedData, subsetCategory = self.selectData(data, QoU)
        return subsetCategory, QoU