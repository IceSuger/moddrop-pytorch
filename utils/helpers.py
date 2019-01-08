
# v3.0.2 为单个低质量多模态样本确定其最优输入模态组合 deltaStar
import itertools
import os
import shutil

import torch

from data_quality.DamageFunctions import Noise


def testExistAndCreateDir(s):
    path = os.path.join(os.getcwd(), s)
    if not os.path.isdir(path):
        # os.mkdir(path)
        os.makedirs(path)

def testExistAndRemoveDir(s):
    path = os.path.join(os.getcwd(), s)
    if os.path.isdir(path):
        # os.mkdir(path)
        shutil.rmtree(path)

def getSubsets(features):
    """
    features: list
    """
    subsets = []
    for i in range(1, len(features) + 1):
        curList = list(map(list, itertools.combinations(features, i)))
        subsets.extend(curList)
    return subsets

noise = Noise(randomly=False)    # randomly = True, 则表示以随机程度加入各类噪声, 这里置否是为了让maskNoise能够以 rate=0 来遮蔽原始数据。
                                 # 后面会再次创建另一个 Noise 实例，其用于随机加噪声，故其 randomly = True 。
def mask(sample, subset):
    '''
    借用 Noise 类中的 mask 噪声实现对整帧的屏蔽
    :param sample:
    :param subset:
    :return:
    '''
    sample = sample.copy()
    for key in sample.keys():
        if key not in subset:
            sample[key] = noise.MaskingNoise(sample[key], 1).float() # .astype(np.float32)
    return sample


def getDeltaStarForOneSample(M0, H, _s, label):
    label = label.data.cpu().numpy()

    Set_probs = []
    modalities = _s.keys()
    Set_modality = getSubsets(list(modalities))

    # # DEBUG
    # print(f'_s type: ')

    # # DEBUG
    # for mdlt in _s.keys():
    #     _s[mdlt] = _s[mdlt].unsqueeze(0)

    for subset in Set_modality:
        probs = M0.model(mask(_s, subset))  # probs 即为模型输出的 score（见 basicClassifier.py 中 test 相关方法）
        result = torch.argmax(probs.data, dim=1)
        Set_probs.append((probs.data.cpu().numpy()[0], int(label), int(result), subset))
        # v2.7 同时记下各种模态组合对应的预测结果和熵值

    Set_probs.sort(key=lambda x: (x[1] != x[2], H(x[0])))

    # print(f'Set_probs = {Set_probs}')

    if Set_probs[0][1] != Set_probs[0][2]:  # 如果排在第一位的这个，预测结果都和label不同，说明所有的结果都是错的，那么就取全集作为 delta_star
        # subset_best = list(modalities).copy()
        subset_best = []
    else:
        subset_best = Set_probs[0][3]

    return subset_best