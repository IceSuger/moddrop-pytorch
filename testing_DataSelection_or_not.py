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
from CONSTS import R
from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal
from datasetsOfLowQualityData.datasetSelectedMultimodal import DatasetSelectedMultimodal
from lqMultimodalClassifier import lqMultimodalClassifier

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

# source_folder = '/mnt/data/dramacha/data_preprocessedv2/'
source_folder = '/home/xiaoyunlong/code/moddrop-pytorch/LowQuality_' + str(R) + '_times/'
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
               'LQ_multimodal': lqMultimodalClassifier,
               'selected_LQ_multimodal': lqMultimodalClassifier
              }

dataset_types = {
            'video' : DatasetVideoClassifier,
            'videoFeat': DatasetVideoFeatureExtractor,
            'skeleton': DatasetSkeleton,
            'audio': DatasetAudio,
            'multimodal': DatasetMultimodal,
            'LQ_multimodal': DatasetOfDamagedMultimodal,
            'selected_LQ_multimodal': DatasetSelectedMultimodal
            }


def commonPartOfTheTesting(cl_mode, step = 4, clf = None, df = None):
    """
    两种模式（带和不带数据选择模块）下的测试流程的公共部分。
    :param cl_mode: 数据集的模态
    :return:
    """

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
    res = classifier.test_torch(dataset_type_cls, phi_s=clf, df=df)

    return res

def testWithoutDataSelection():
    cl_mode = 'LQ_multimodal'
    return commonPartOfTheTesting(cl_mode)

def testWithDataSelection(clf, df):
    """

    :param clf: 用于完成数据选择的分类器 phi_s
    :param df: 保存了 QoU->delta 的表，且其中的 df.cc.cat.categories 为根据输出的数字返回相应的模态子集的映射表
    :return:
    """
    cl_mode = 'selected_LQ_multimodal'
    # print(f'clf = {clf}')
    # print(f'df = {df}')

    return commonPartOfTheTesting(cl_mode, clf=clf, df=df)
