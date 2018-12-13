import numpy as np

class DataQuality():
    def getMetricFuncs(self):
        return [self.std,
                self.SNR,
                self.integrity,
                self.zeroRate]
    '''

    就 Chalearn 2014 数据集而言，输入 X 都是单个模态的 stblock。
    '''
    def std(self, X):
        '''
        标准差
        '''
        # print(f'In std QoU func, type(X)={type(X)}, X.shape={X.shape}')
        return np.std(X, ddof=1) # ddof = 1, 计算的是样本标准差，该参数默认为0，计算总体标准差

    def SNR(self, X):
        '''
        信噪比
        '''
        std = np.std(X)
        if std is None or std == 0 or np.mean(X) is None:
            return 0    # 非法时直接返回0，不知道合不合理。暂时这么做吧，反正数据评价这块，目前整体都不太靠谱。

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

        [2018-9-17 v1.2] 实际上这里的输入是 (1, 4, 5, 1, 36, 36) 这样的，应该遍历shape[2]
        '''
        missingCnt = 0
        # X = X.unsqueeze(0) # [2018-12-6 v2.0] 不知道为啥，第一个维度的1没了....自己手动补上来吧先。
        shape = X.shape

        # print(f"X is: {X}")
        # print(f"X shape is {X.shape}, shape[1] = {shape[1]}")
        # print(f"X[:] shape is {X[:].shape}")
        # print(f"X[:][:] shape is {X[:][:].shape}")

        X = X.copy()
        X = np.reshape(X, (shape[0]*shape[1], -1))

        # print(f"X shape is {X.shape}, shape[1] = {shape[1]}")

        for frame in range(shape[1]): # [2018-12-6 v2.0] 不知道为啥，第一个维度的1没了....
            # print(f'Now, frame = {frame} !!!')
        # for frame in range(shape[1]):
        #     fr = X[:][:][frame]
            fr = X[frame]
            if np.max(fr) == 0 and  np.min(fr) == 0:
                missingCnt += 1
        score = missingCnt / shape[1]
        return score

    def zeroRate(self, X):
        '''
        零值占比
        :param X:
        :return:
        '''
        zeroCnt = ((X == 0) * 1).sum()
        return zeroCnt / X.size



'''
越界值占比
'''

'''
异常值占比
'''

