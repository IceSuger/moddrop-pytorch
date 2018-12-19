"""
1. generate damaged multimodal dataset and QoUs over this dataset
2. generate labels
"""
import shutil

import pandas as pd

from CONSTS import EXPERIMENT_RESULT_FILE_NAME, LQDataset_on_subset_with_dmgfunc_at_degree_ROOT, PREDICTED_DELTAS_PATH
from datasetsOfLowQualityData.datasetDmgedMultimodalSelectedDelta import DatasetDmgedMultimodalSelectedDelta
from datasetsOfLowQualityData.datasetOfDmgedMultimodalAndQoU import DatasetOfDamagedMultimodalAndQoU
from lqMultimodalClassifier import lqMultimodalClassifier
from moddropMultimodalClassifier import moddropMultimodalClassifier
from multimodalClassifier import multimodalClassifier
from datasets.datasetMultimodal import DatasetMultimodal
from data_quality.QualityMetric import DataQuality
from data_quality.DamageFunctions import Noise
from entropyEvaluating import entropyOnProbs

from torch.utils.data import DataLoader
import pickle
import os
import itertools
import torch

from testing_DataSelection_or_not import testWithoutDataSelection, testWithDataSelection


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

source_folder = '/home/xiaoyunlong/downloads/DeepGesture/Montalbano/'
'''
	Location of the dataset (Chalearn 2014) which has been pre-processed.
'''

filter_folder = 'filters/'
classifier = multimodalClassifier(step = 4,
                                 input_folder = source_folder,
                                 filter_folder = filter_folder)

# 1. generate damaged multimodal dataset and QoUs over this dataset

"""
for s in D0:
    for u in range(r):  # randomly get r damaged data sample
        _s = {}
        QoU = []
        for mdlt in s:
            for dmg_dimension in dmg_functions:
                _s[mdlt] = dmg_func(s[mdlt])

            for score_func in score_funcs:
                QoU.append(score_func(_s[mdlt]))

        # save to disk
        cPickle(_s, QoU, label_of_s, u, if_s_name_available_then_set_it_here)
"""
def generateLQDataset(r = 8, subset = 'train'):
    # # Consts
    # r = 8   # 单条原多模态数据，生成 r 条被破坏记录
    #

    train_data = DatasetMultimodal(classifier.input_folder, 'mocap', subset, classifier.hand_list,
                                   classifier.seq_per_class,
                                   classifier.nclasses, classifier.input_size, classifier.step, classifier.nframes)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=12)

    dmg_functions = Noise(randomly=True).getDmgFunctions()
    score_functions = DataQuality().getMetricFuncs()

    n = len(train_loader)
    print(f'len(train_loader) = {len(train_loader)}')
    for ii, (data, label) in enumerate(train_loader):

        if ii % 100 == 0:
            print(f'ii: {ii}, {ii/n}')

        for u in range(r):
            _s = {}
            QoU = []
            for mdlt in data.keys():
                for dmg_func in dmg_functions:
                    _s[mdlt] = dmg_func(data[mdlt].data.numpy()[0])

                for score_func in score_functions:
                    QoU.append(score_func(_s[mdlt]))

            # save to disk
            damaged_multimodal = {}
            damaged_multimodal['data'] = _s
            damaged_multimodal['QoU'] = QoU
            damaged_multimodal['label'] = label # 这里的label，是原始多模态任务中的label，如手势类别
            filename = str(label.data) + '_' + str(ii) + '_' + str(u)
            # testExistAndCreateDir('damaged_multimodal/')
            # pickle.dump(damaged_multimodal, open('damaged_multimodal/' + filename, 'wb'))
            path = 'LowQuality_'+str(r)+'_times/' + subset + '/'
            testExistAndCreateDir(path) # 原始高质量数据集的r倍数量的样本数的低质量数据
            pickle.dump(damaged_multimodal, open(path + filename, 'wb'))


def generateLQDataset_for_experiment1(r = 8, subset = 'valid', clf = None, df = None):
    # # Consts
    # r = 8   # 单条原多模态数据，生成 r 条被破坏记录
    """
    【注意】phi_s 是用各种随机的破坏程度得到的各种各样的质量评分向量对应的 D_Q 训练出来的！且是固定的。

    遍历被污染模态子集 delta：
        遍历破坏方式 dmg_func in dmg_funcs：
            遍历破坏程度 degree = 0, 10, 20, ..., 100 %:
                            2.1 LQ_dataset = dmg_func(delta, degree, HQ_dataset)
                            2.2 不经过数据选择模块，跑 test
                            2.3 经过数据选择模块，跑 test
    """

    # 预先准备好需要的东西
    train_data = DatasetMultimodal(classifier.input_folder, 'mocap', subset, classifier.hand_list,
                                   classifier.seq_per_class,
                                   classifier.nclasses, classifier.input_size, classifier.step, classifier.nframes)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=12)
    _s, _ = train_data.__getitem__(0)  # 随便取一个样本，主要是为了取其模态集合，为了下面遍历其幂集

    modalities = _s.keys()
    Set_modality = getSubsets(list(modalities))

    dmg_functions = Noise().getDmgFunctions()

    res_file_name = EXPERIMENT_RESULT_FILE_NAME

    # 开始
    delta_cnt = 0
    for delta in Set_modality:
        delta_cnt += 1
        dmg_func_cnt = 0
        for dmg_func in dmg_functions:
            dmg_func_cnt += 1
            degree_cnt = 0
            for degree in range(1, 11):
                degree_cnt += 1
                degree *= 0.1

                # 2.0 先清空现有数据
                testExistAndRemoveDir(LQDataset_on_subset_with_dmgfunc_at_degree_ROOT)

                # 2.1
                # 一次性的，生成的数据跑一遍测试，就删掉。都存在同一个路径。
                generateLQDataset_on_subset_with_dmgfunc_at_degree(r, dmg_func, degree, delta, train_loader)

                # 2.2
                accuracy_no_data_selection = testWithoutDataSelection()

                # 2.3
                accuracy_with_data_selection = testWithDataSelection(clf, df)

                # 2.4
                # 将结果写到文件
                res_file = open(res_file_name, 'a')
                # res_file.write(f'delta:{delta}\tdmg_func:{dmg_func.__name__}\tdegree:{degree}, accuracy_with_data_selection:{accuracy_with_data_selection}, \taccuracy_no_data_selection:{accuracy_no_data_selection}\n')
                res_file.write(
                    f'{delta}\t{dmg_func.__name__}\t{degree}\t{accuracy_with_data_selection}\t{accuracy_no_data_selection}\n')
                res_file.close()

                # 2.5
                # 汇报一下进度
                print(f'FINISHED: dmg_func:{dmg_func_cnt} / {len(dmg_functions)}, degree:{degree_cnt} / 10, delta:{delta_cnt} / {len(Set_modality)}')

def generateLQDataset_on_subset_with_dmgfunc_at_degree(r, dmg_func, degree, delta, data_loader):

    score_functions = DataQuality().getMetricFuncs()

    n = len(data_loader)
    for ii, (data, label) in enumerate(data_loader):
        if ii % 100 == 0:
            print(f'dmg_func:{dmg_func.__name__}\tdegree:{degree}\tdelta:{delta}, \tii: {ii}, {ii/n}')

        for u in range(r):
            _s = {}  # data.copy()
            QoU = []

            # 先破坏，再评分。破坏是对delta对应的模态
            for mdlt in data.keys():
                if mdlt in delta:
                    _s[mdlt] = dmg_func(X = data[mdlt].data.numpy()[0],  rate = degree)
                else:
                    _s[mdlt] = data[mdlt].data.numpy()[0]

            # 先破坏，再评分。评分是对所有模态，破坏是对delta对应的模态
            for mdlt in data.keys():
                for score_func in score_functions:
                    QoU.append(score_func(_s[mdlt]))

            # save to disk
            damaged_multimodal = {}
            damaged_multimodal['data'] = _s
            damaged_multimodal['QoU'] = QoU
            damaged_multimodal['label'] = label # 这里的label，是原始多模态任务中的label，如手势类别
            filename = str(label.data) + '_' + str(ii) + '_' + str(u) + '_' + dmg_func.__name__ + '_' + str(degree) + "'".join(delta)

            # path = 'LowQuality_'+str(r)+'_times/' + subset + '/'
            path = LQDataset_on_subset_with_dmgfunc_at_degree_ROOT + 'valid' + '/'

            testExistAndCreateDir(path) # 原始高质量数据集的r倍数量的样本数的低质量数据
            pickle.dump(damaged_multimodal, open(path + filename, 'wb'))


def predictDeltas(clf, df, dmg_func, degree, delta):
    """
    由 DatasetDmgedMultimodalSelectedDelta 读取一遍新生成的 valid set，从而得到相应 valid set 上样本们对应的模态筛选子集 delta 们，并存入相应的 csv 文件
    :param clf:
    :param df:
    :param dmg_func: 用于命名结果csv文件
    :param degree: 用于命名结果csv文件
    :param delta: 用于命名结果csv文件
    :return:
    """
    '''
    1. DatasetDmgedMultimodalSelectedDelta(self.input_folder, train_valid_test='valid', phi_s=phi_s, QoU2delta_df=df)
    2. 遍历dataset，取item存df
    3. 保存df到csv
    '''
    # 1
    dataset = DatasetDmgedMultimodalSelectedDelta(LQDataset_on_subset_with_dmgfunc_at_degree_ROOT, train_valid_test='valid', phi_s=clf, QoU2delta_df=df)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2
    df_list = []
    n = len(data_loader)
    for ii, (subsetCategory, QoU) in enumerate(data_loader):
        print(
            f'dmg_func:{dmg_func.__name__}\tdegree:{degree}\tdelta:{delta}, \tii: {ii}, {ii/n}')

        df_list.append([subsetCategory, QoU])

    # 3
    filename = f'{dmg_func.__name__}-{degree}-{delta}.csv'
    root = PREDICTED_DELTAS_PATH
    testExistAndCreateDir(root)

    df = pd.DataFrame(df_list)
    df.to_csv(root+filename, sep='\t', header=False, index=False)


def generateLQDataset_for_experiment2(r = 8, subset = 'valid', clf = None, df = None):
    """
    这个方法是为了看：(dmg_func, dmg_degree, dmg_delta) 对应生成的 valid set 上，phi_s 预测出的模态选择 delta 的分布情况。
    看看是否理想，以及不理想的话怎么调整。
    :param r:
    :param subset:
    :param clf:
    :param df:
    :return:
    """
    # # Consts
    # r = 8   # 单条原多模态数据，生成 r 条被破坏记录
    """
    【注意】phi_s 是用各种随机的破坏程度得到的各种各样的质量评分向量对应的 D_Q 训练出来的！且是固定的。

    遍历破坏方式 dmg_func in dmg_funcs：
            遍历破坏程度 degree = 0, 10, 20, ..., 100:
                    遍历被污染模态子集 delta：
                            2.1 LQ_dataset = dmg_func(delta, degree, HQ_dataset)
                            2.2 不经过数据选择模块，跑 test
                            2.3 经过数据选择模块，跑 test
    """

    # 预先准备好需要的东西
    train_data = DatasetMultimodal(classifier.input_folder, '', subset, classifier.hand_list,
                                   classifier.seq_per_class,
                                   classifier.nclasses, classifier.input_size, classifier.step, classifier.nframes)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=56)
    _s, _ = train_data.__getitem__(0)  # 随便取一个样本，主要是为了取其模态集合，为了下面遍历其幂集

    modalities = _s.keys()
    Set_modality = getSubsets(list(modalities))

    dmg_functions = Noise().getDmgFunctions()

    res_file_name = 'experiment1_(v2.4)_results.txt'

    # 开始
    delta_cnt = 0
    for delta in Set_modality:
        delta_cnt += 1
        dmg_func_cnt = 0
        for dmg_func in dmg_functions:
            dmg_func_cnt += 1
            degree_cnt = 0
            for degree in range(1, 11):
                degree_cnt += 1
                degree *= 0.1

                # 2.0 先清空现有数据
                testExistAndRemoveDir(LQDataset_on_subset_with_dmgfunc_at_degree_ROOT)

                # 2.1
                # 一次性的，生成的数据跑一遍测试，就删掉。都存在同一个路径。
                generateLQDataset_on_subset_with_dmgfunc_at_degree(r, dmg_func, degree, delta, train_loader)

                # 3.0
                # 由 DatasetDmgedMultimodalSelectedDelta 读取一遍新生成的 valid set，从而得到相应 valid set 上样本们对应的模态筛选子集 delta 们，并存入相应的 csv 文件
                predictDeltas(clf, df, dmg_func, degree, delta)

                # 3.5
                # 汇报一下进度
                print(f'FINISHED: dmg_func:{dmg_func_cnt} / {len(dmg_functions)}, degree:{degree_cnt} / 10, delta:{delta_cnt} / {len(Set_modality)}')


# 2. generate labels

"""
for _s, QoUs in DataLoader(dataset_generated_above):
    # s_probs = set()
    Set_probs = []
    Set_modality = getSubsets([modalities])
    for subset in Set_modality:
        probs, result = M0(mask(_s, subset))
        Set_probs.append((probs, result, label, subset))

        #if result correct:  # 只有分类正确的前提下，熵低才有意义
               # 那么存在一个情况，可能分类就都不正确啊...

    sort Set_probs by (-correct, H(probs)) # ascend
    # 取第一个

    # if Set_probs is empty:
    #     subset_best =
    # else:
    #     subset_best = argminH(Set_probs)

    # save to disk
    cPickle(QoUs, subset_best)

    # or save in memory
    df(dtypes = {'subset_best': str, others: float})
"""
def generateDeltaStar(r = 8, train_valid_test = 'train', path_D_Q_root = None):
    path = 'LowQuality_' + str(r) + '_times/'# + train_valid_test + '/'
    path_D_Q = path_D_Q_root + '/' + train_valid_test + '/'

    # dataset_damaged_multimodal = DatasetOfDamagedMultimodal(os.path.join(os.getcwd(), 'damaged_multimodal/'))
    dataset_damaged_multimodal_and_qou = DatasetOfDamagedMultimodalAndQoU(os.path.join(os.getcwd(), path), train_valid_test)
    dataset_generated_above = DataLoader(dataset_damaged_multimodal_and_qou, batch_size=1, shuffle=False, num_workers=8)

    # 预备下面要用的方法
    # [Xiao] v2.5 这里的分类器不再是 lqMultimodalClassifier 了！！！改成了 moddropMultimodalClassifier ！！！！！
    M0 = moddropMultimodalClassifier(step = 4,
                                     input_folder = source_folder,
                                     filter_folder = filter_folder)
    H = entropyOnProbs  # 根据概率向量求熵

    # 预备好M0，加载已训练的权重，进入预测模式（非训练模式，一方面保证权重不更新，一方面保证batchnorm可以在batchsize为1时正常工作）并挪上GPU
    M0.build_network() # build the model
    M0.load_weights()
    M0.model.eval()
    M0.model.to(M0.device)

    print(f'dataset_generated_above length = {len(dataset_generated_above)}')

    with torch.no_grad():
        for ii, (_s, label, QoU) in enumerate(dataset_generated_above):
            label = label.data.cpu().numpy()

            n = len(dataset_generated_above)
            if ii % 100 == 0:
                print(f'ii: {ii}, {ii/n}')

            Set_probs = []
            modalities = _s.keys()
            Set_modality = getSubsets(list(modalities))

            # # DEBUG
            # print(f'_s type: ')

            # # DEBUG
            # for mdlt in _s.keys():
            #     _s[mdlt] = _s[mdlt].unsqueeze(0)

            for subset in Set_modality:
                probs = M0.model(mask(_s, subset))    # probs 即为模型输出的 score（见 basicClassifier.py 中 test 相关方法）
                result = torch.argmax(probs.data, dim=1)
                Set_probs.append((probs.data.cpu().numpy()[0], int(label), int(result), subset))
                # v2.7 同时记下各种模态组合对应的预测结果和熵值


            Set_probs.sort(key=lambda x: (x[1]!=x[2], H(x[0])))

            # print(f'Set_probs = {Set_probs}')

            if Set_probs[0][1] != Set_probs[0][2]:  # 如果排在第一位的这个，预测结果都和label不同，说明所有的结果都是错的，那么就取全集作为 delta_star
                # subset_best = list(modalities).copy()
                subset_best = []
            else:
                subset_best = Set_probs[0][3]

            sample_for_M1 = {}
            sample_for_M1['QoU'] = QoU#.data.cpu().numpy()[0]
            sample_for_M1['subset_best'] = subset_best
            filename = str(label) + '_' + str(ii)
            # testExistAndCreateDir('train_for_M1/')
            # pickle.dump(sample_for_M1, open('train_for_M1/' + filename, 'wb'))
            testExistAndCreateDir(path_D_Q)
            pickle.dump(sample_for_M1, open(path_D_Q + filename, 'wb'))

            # # v2.7(?)-暂时跳过这个版本的实现。
            #  同时记下各种模态组合对应的预测结果和熵值，以及本来的正确类别，还有QoU、delta_star
            # list_to_save.append(QoU.extend([]))



# 3. Train M1
#
# M1 实现为 nn.Module 子类，两层神经网络
"""

"""


# 4. Use M1
#
# 输入
#   s: dict, 多模态数据
# 输出
#   result: 手势识别结果（类别）
"""
QoU = []
for mdlt in s:
    for score_func in score_funcs:
        QoU.append(score_func(_s[mdlt]))
subset_best = M1(QoU)
masked_s = mask(s, subset_best)    # subset: list, s: dict(key: mdlt)

result = M0(masked_s)
"""