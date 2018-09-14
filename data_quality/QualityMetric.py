import numpy as np

class DataQuality():
    '''

    就 Chalearn 2014 数据集而言，输入 X 都是单个模态的 stblock。
    '''
    def std(self, X):
        '''
        标准差
        '''
        return np.std(X, ddof=1) # ddof = 1, 计算的是样本标准差，该参数默认为0，计算总体标准差

    def SNR(self, X):
        '''
        信噪比
        '''
        return np.mean(X) / np.std(X)

    def integrity(self, X):
        '''
        完整性

        插值可以补全数据。但是插值应该是影响结果的。怎么衡量这个完整性呢。

        破坏完整性：通过MaskNoise来把整帧都干掉。

        这样吧。破坏都是针对单帧破坏。然后针对单帧打分，再合并得到整个 5 帧 stblock 的评分。


        输入：
        X: numpy.ndarray 例如尺寸 (5, 36, 36)
        如果某帧缺失，则相应帧的所有像素值应该全为 0

        '''
        missingCnt = 0
        shape = X.shape
        for frame in shape[0]:
            fr = X[frame]
            if np.max(fr) == 0 and  np.min(fr) == 0:
                missingCnt += 1
        score = missingCnt / shape[0]
        return score





'''
越界值占比
'''

'''
异常值占比
'''

