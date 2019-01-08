#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
from CONSTS import R, LQDataset_on_subset_with_dmgfunc_at_degree_ROOT
from datasetsOfLowQualityData.datasetModdropMultimodal import DatasetModdropMultimodal
from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal
from datasetsOfLowQualityData.datasetSelectedMultimodal import DatasetSelectedMultimodal
from lqMultimodalClassifier import lqMultimodalClassifier
from moddropMultimodalClassifier import moddropMultimodalClassifier

__docformat__ = 'restructedtext en'

import os

os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# from motionDetector import motionDetector
from audioClassifier import audioClassifier
from skeletonClassifier import skeletonClassifier
from videoFeatureExtractor import videoFeatureExtractor
from videoClassifier import videoClassifier
from multimodalClassifier import multimodalClassifier

from datasets.datasetVideoClassifier import DatasetVideoClassifier
from datasets.datasetVideoFeatureExtractor import DatasetVideoFeatureExtractor
from datasets.datasetSkeleton import DatasetSkeleton
from datasets.datasetAudio import DatasetAudio
from datasets.datasetMultimodal import DatasetMultimodal

''' Import the relevant classes from the util module.
	 - skeletonClassifier trains a 3D-ConvNet using mocap data
	 - videoClassifier trains a 3D convNet using video data of hands
	 - multimodalClassifier trains using both moCap and video data
	 - motionDetector trains a motion detector and used in post-training
	   to improve classification results.
	   This classifier implements late-fusion using a shared-hidden
	   layer
'''
import torch
torch.manual_seed(1) # 设定随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

torch.backends.cudnn.deterministic=True

source_data_folder = '/home/xiaoyunlong/code/moddrop-pytorch/LowQuality_' + str(R) + '_times/'     # For training with LQ dataset
# source_folder = '/home/xiaoyunlong/downloads/DeepGesture/Montalbano/'                           # For training with HQ dataset

'''
	Location of the dataset (Chalearn 2014) which has been pre-processed.
'''

filter_folder = 'filters/'

'''
	Filters refer to the saved-weights of the pre-trained
	convolutional neural network
'''

cl_methods = { 'skeleton' : skeletonClassifier,
              # 'motionDetector' : motionDetector,
              'video' : videoClassifier,
              'videoFeat': videoFeatureExtractor,
              'audio' : audioClassifier,
              'multimodal' : multimodalClassifier,
               'LQ_multimodal': moddropMultimodalClassifier,    # v2.5.3.1 改过来的
               'selected_LQ_multimodal': moddropMultimodalClassifier, # v2.5.3.1 改过来的

                'moddropMultimodal': moddropMultimodalClassifier
              }

dataset_types = {
            'video' : DatasetVideoClassifier,
            'videoFeat': DatasetVideoFeatureExtractor,
            'skeleton': DatasetSkeleton,
            'audio': DatasetAudio,
            'multimodal': DatasetMultimodal,
            'LQ_multimodal': DatasetOfDamagedMultimodal,
            'selected_LQ_multimodal': DatasetSelectedMultimodal,

            'moddropMultimodal': DatasetModdropMultimodal
            }


def commonPartOfTheTesting(cl_mode, step = 4, clf = None, df = None, tuple_M0_and_H = None, source_folder = None):
    """
    两种模式（带和不带数据选择模块）下的测试流程的公共部分。
    :param cl_mode: 数据集的模态
    :return:
    """
    if source_folder is None:   # 如果不传入这个参数
        source_folder = source_data_folder  # 就用该文件开头声明的变量

    #try:
    classifier = cl_methods[cl_mode](step = step,
                                     input_folder = source_folder,
                                     filter_folder = filter_folder)#,
                                     # pretrained = True) # 设为True，从而在初始化网络时，自动加载权重
    dataset_type_cls = dataset_types[cl_mode]

    classifier.build_network() # build the model

    classifier.load_weights()
    # classifier.train_torch(dataset_type_cls)
    # print(f'In testing_DataSelection_or_not.py, df = {df}')
    if not tuple_M0_and_H is None:
        M0 = tuple_M0_and_H[0]
        H = tuple_M0_and_H[1]
    else:
        M0 = None
        H = None
    res = classifier.test_torch(dataset_type_cls, phi_s=clf, df=df, M0=M0, H=H)

    return res

def testHQWithoutDataSelection():
    # cl_mode = 'multimodal'
    cl_mode = 'moddropMultimodal'
    return commonPartOfTheTesting(cl_mode, source_folder='/home/xiaoyunlong/downloads/DeepGesture/Montalbano/')

def testWithoutDataSelection():
    cl_mode = 'LQ_multimodal'
    return commonPartOfTheTesting(cl_mode, source_folder=LQDataset_on_subset_with_dmgfunc_at_degree_ROOT)

def testWithDataSelection(clf, df, tuple_M0_and_H):
    """

    :param clf: 用于完成数据选择的分类器 phi_s
    :param df: 保存了 QoU->delta 的表，且其中的 df.cc.cat.categories 为根据输出的数字返回相应的模态子集的映射表
    :return:
    """
    cl_mode = 'selected_LQ_multimodal'
    # print(f'clf = {clf}')
    # print(f'df = {df}')

    return commonPartOfTheTesting(cl_mode, clf=clf, df=df, tuple_M0_and_H=tuple_M0_and_H, source_folder=LQDataset_on_subset_with_dmgfunc_at_degree_ROOT)

