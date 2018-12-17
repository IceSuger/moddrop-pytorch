from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal
import numpy as np

class DatasetSelectedMultimodal(DatasetOfDamagedMultimodal):
    def __init__(self, root, phi_s, train_valid_test='train', QoU2delta_df=None):
        """
        :type subset: string
        :param subset: string representing 'train', 'validation' or 'test' subsets
        """
        super().__init__(root, train_valid_test=train_valid_test)
        self.phi_s = phi_s
        # self.file_list_len = len(self.file_list)
        self.QoU2delta_df = QoU2delta_df
        # print(f'QoU2delta_df.head() = {QoU2delta_df.head()}')
        self.table_susbetCode_to_subsetCategory = self.init_table_susbetCode_to_subsetCategory(QoU2delta_df.cc.cat.categories)


    def __getitem__(self, ind):
        data, label, QoU = self._my_getitem__(ind)
        selectedData, _ = self.selectData(data, QoU)
        return selectedData, label

    # def __len__(self):
    #     return self.file_list_len

    def init_table_susbetCode_to_subsetCategory(self, categories):
        tmpList = categories.tolist()
        resList = []
        for item in tmpList:
            resList.append(item.replace("'","").split(', '))
        return resList

    def selectData(self, data, QoU):
        """
        依据多模态样本 data 对应的质量描述向量 QoU ，由已训练的数据选择分类器 phi_s 完成数据选择
        :param phi_s: 已训练的分类器
        :param data: {'color':XXX, 'depth':XXX, 'audio':XXX, ...}
        :param QoU: DataFrame里的一行？
        :return:
        """
        # print(data)
        selectedData = data.copy()
        # print(type(data))
        # print(f'QoU = {QoU}')
        """
        1. 输入 QoU 到 phi_s，得到 delta_star;
        2. 查表（需要加载 df.cc.cat.categories 进来），得到 delta_star 对应的模态集合；
        3. 将 data 中不属于这个集合的模态，都乘以0，结果赋值给 selectedData；
        4. 返回 selectedData
        """
        # 1.
        # print(f'QoU={QoU}')
        subsetCode = self.phi_s.predict(np.array(QoU).reshape(1, -1))[0]
        # print(f'subsetCode = {subsetCode}')
        # 2.
        # print(f'self.table_susbetCode_to_subsetCategory = {self.table_susbetCode_to_subsetCategory}')
        subsetCategory = self.table_susbetCode_to_subsetCategory[subsetCode]

        # 3.
        # print(f'subsetCategory={subsetCategory} \t len(subsetCategory)={len(subsetCategory)} \t type(subsetCategory)={type(subsetCategory)}')
        # # if len(subsetCategory) == 0:
        # if not subsetCategory or not subsetCategory[0]:
        #     print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

        # 注意，如果预测出来的 delta 为空（或者空list），则其实对应了原来训练 phi_s 的 D_Q 里的不能被 phi_r 正确分类的样本，这种情况下，我们这里选择【保留全部模态】
        if subsetCategory and subsetCategory[0]:
            for mdlt in selectedData.keys():
                 if not mdlt in subsetCategory:
                    selectedData[mdlt] = 0 * selectedData[mdlt]
        # 4.
        return selectedData, subsetCategory