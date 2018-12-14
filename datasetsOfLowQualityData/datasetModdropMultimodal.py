from CONSTS import MODDROP_RATE
from datasets.datasetMultimodal import DatasetMultimodal
import numpy as np

class DatasetModdropMultimodal(DatasetMultimodal):
    def __init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes):
        DatasetMultimodal.__init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes)

    def __getitem__(self, ind):
        data, label = DatasetMultimodal.__getitem__(self, ind)
        selectedData = data.copy()

        # 随机屏蔽部分模态，subsetCategory 为最重要保留的模态
        subsetCategory = self.generateSubset(selectedData.keys(), rate=MODDROP_RATE)

        # 屏蔽要被屏蔽的模态。将 不属于 subsetCategory 的模态，给置为 0
        for mdlt in selectedData.keys():
            if not mdlt in subsetCategory:
                selectedData[mdlt] = 0 * selectedData[mdlt]

        return selectedData, label

    def generateSubset(self, mdlts, rate):
        """

        :param mdlts:
        :param rate: 每个模态被丢弃的概率
        :return:
        """
        res = []
        for mdlt in mdlts:
            # 从 [0,1) 的均匀分布中取 1 个值
            if not np.random.uniform(0,1) < rate:
                res.append(mdlt)
        # 如果结果是 res 里为空，显然不行。那就在这种情况下返回全集。
        if not res:
            res = list(mdlts)

        return res.copy()