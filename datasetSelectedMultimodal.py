from datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal


def selectData(phi_s, data, QoU):
    """
    依据多模态样本 data 对应的质量描述向量 QoU ，由已训练的数据选择分类器 phi_s 完成数据选择
    :param phi_s: 已训练的分类器
    :param data: {'color':XXX, 'depth':XXX, 'audio':XXX, ...}
    :param QoU: DataFrame里的一行？
    :return:
    """
    print(data)
    print(type(data))


class DatasetSelectedMultimodal(DatasetOfDamagedMultimodal):
    def __init__(self, root, phi_s):
        """
        :type subset: string
        :param subset: string representing 'train', 'validation' or 'test' subsets
        """
        super().__init__(root)
        self.phi_s = phi_s
        self.file_list_len = len(super().file_list)


    def __getitem__(self, ind):
        """


        :param index:
        :return:
        """
        # filepath = self.file_list[ind]
        # with open(filepath, 'rb') as f:
        #     sample = pickle.load(f, encoding='iso-8859-1')
        #     # 读取的值默认是double型，这里改成float型，否则后面送进模型时会出错。
        #     for mdlt in sample['data'].keys():
        #         sample['data'][mdlt] = sample['data'][mdlt].astype(numpy.float32)
        # # _s, label, QoU
        # return sample['data'], sample['label'], sample['QoU']

        data, label, QoU = super()._my_getitem__(ind)
        selectedData = selectData(self.phi_s, data, QoU)
        return selectedData, label

    def __len__(self):
        return self.file_list_len