from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal

class DatasetOfDamagedMultimodalAndQoU(DatasetOfDamagedMultimodal):
    def __init__(self, root, train_valid_test='train'):
        """
        :type subset: string
        :param subset: string representing 'train', 'validation' or 'test' subsets
        """
        super().__init__(root, train_valid_test)
        self.file_list_len = len(self.file_list)


    def __getitem__(self, ind):
        return super()._my_getitem__(ind)

    def __len__(self):
        return self.file_list_len