"""
1. generate damaged multimodal dataset and QoUs over this dataset
2. generate labels
"""
from multimodalClassifier import multimodalClassifier
from datasetMultimodal import DatasetMultimodal
from datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal

from torch.utils.data import DataLoader
import pickle
import os

def testExistAndCreateDir(s):
    path = os.path.join(os.getcwd(), s)
    if not os.path.isdir(path):
        os.mkdir(path)

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
# Consts
r = 8


train_data = DatasetMultimodal(classifier.input_folder, '', 'train', classifier.hand_list,
                               classifier.seq_per_class,
                               classifier.nclasses, classifier.input_size, classifier.step, classifier.nframes)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=56)

for ii, (data, label) in enumerate(train_loader):
    for u in range(r):
        _s = {}
        QoU = []
        for mdlt in data.keys():
            for dmg_func in dmg_functions:
                _s[mdlt] = dmg_func(data[mdlt])

            for score_func in score_functions:
                QoU.append(score_func(_s[mdlt]))

        # save to disk
        damaged_multimodal = {}
        damaged_multimodal['data'] = _s
        damaged_multimodal['QoU'] = QoU
        damaged_multimodal['label'] = label
        filename = label + '_' + str(ii) + '_' + str(u)
        testExistAndCreateDir('damaged_multimodal/')
        pickle.dump(damaged_multimodal, open('damaged_multimodal/' + filename, 'wb'))


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
dataset_damaged_multimodal = DatasetOfDamagedMultimodal()
dataset_generated_above = DataLoader(dataset_damaged_multimodal)

for ii, (_s, label, QoU) in enumerate(dataset_generated_above):
    Set_probs = []
    modalities = _s.keys()
    Set_modality = getSubsets([modalities])

    for subset in Set_modality:
        probs, result = M0(mask(_s, subset))
        Set_probs.append((probs, label, result, subset))

    Set_probs.sort(key=lambda x: (x[1]!=x[2], H(x[0])))
    subset_best = Set_probs[0][3]

    sample_for_M1 = {}
    sample_for_M1['QoU'] = QoU
    sample_for_M1['subset_best'] = subset_best
    filename = label + '_' + str(ii)
    testExistAndCreateDir('train_for_M1/')
    pickle.dump(sample_for_M1, open('train_for_M1/' + filename, 'wb'))



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