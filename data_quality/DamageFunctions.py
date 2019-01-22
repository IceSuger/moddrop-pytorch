import numpy as np


'''
椒盐噪声
'''

'''
高斯噪声
'''

'''
删除部分数据
'''

import numpy


class Noise():
    def __init__(self, randomly=False, randomlyOnAndOff=False):
        self.randomly = randomly    # 进一步理论或实验探索这里随机加破坏的程度服从不同分布，是否对结果有影响？
        self.randomlyOnAndOff = randomlyOnAndOff
        self.randomlyOn_prob = 0.5
        self.dmg_functions = [self.SaltAndPepper,
                              self.GaussianNoise,
                              self.MaskingNoise]

    def getDmgFunctions(self):
        return self.dmg_functions

    def SaltAndPepper(self, X, rate=0.3):
        # print(X)
        # print(X.shape)
        # Salt and pepper noise
        if self.randomly:
            # rate = np.random.random()
            rate = np.random.uniform(0, 1)
            # print(f'Noise.randomly is TRUE. In SaltAndPepper, rate={rate}')

        # v3.0.3
        if self.randomlyOnAndOff:
            if np.random.uniform(0, 1) < self.randomlyOn_prob:
                rate = 0

        drop = numpy.random.uniform(0, 1, X.shape)
        z = numpy.where(drop < 0.5 * rate)
        o = numpy.where(numpy.abs(drop - 0.75 * rate) < 0.25 * rate)
        X[z] = 0
        X[o] = 1
        return X

        # drop = numpy.arange(X.shape[1])
        # numpy.random.shuffle(drop)
        # sep = int(len(drop) * rate)
        # drop = drop[:sep]
        # X[:, drop[:sep / 2]] = 0
        # X[:, drop[sep / 2:]] = 1
        # return X

    def GaussianNoise(self, X, rate=None):
        '''

        :param X:
        :param sd: 加入噪声的 std
        :return:
        '''
        # Injecting small gaussian noise
        if self.randomly:
            # rate = np.random.random()
            rate = np.random.uniform(0, 1)

        # v3.0.3
        if self.randomlyOnAndOff:
            if np.random.uniform(0, 1) < self.randomlyOn_prob:
                rate = 0

        if rate is None:
            sd = 0.5
        else:
            sd = rate * np.max(X)

        X += numpy.random.normal(0, sd, X.shape)
        return X

    def MaskingNoise(self, X, rate=0.5):
        if self.randomly:
            # rate = np.random.random()
            rate = np.random.uniform(0, 1)

        # v3.0.3
        if self.randomlyOnAndOff:
            if np.random.uniform(0, 1) < self.randomlyOn_prob:
                rate = 0

        mask = (numpy.random.uniform(0, 1, X.shape) > rate).astype("i4")    # v2.4 之前，这里的大于号被误写为小于号了。所以传入的rate就不再代表被遮蔽的程度，而是原数据被保留的程度
        X = mask * X
        return X

    # def MaskingWholeFrame(self, X, rate=0.5):
    #     if self.randomly:
    #         rate = np.random.random()
    #
    #     mask = (numpy.random.uniform(0, 1, X.shape) < rate).astype("i4")
    #     X = mask * X
    #     return X


def SaltAndPepper(rate=0.3):
    # Salt and pepper noise
    def func(X):
        drop = numpy.random.uniform(0, 1, X.shape)
        z = numpy.where(drop < 0.5 * rate)
        o = numpy.where(numpy.abs(drop - 0.75 * rate) < 0.25 * rate)
        X[z] = 0
        X[o] = 1
        return X

    return func


def GaussianNoise(self, sd=0.5):
    # Injecting small gaussian noise
    def func(X):
        X += numpy.random.normal(0, sd, X.shape)
        return X

    return func
