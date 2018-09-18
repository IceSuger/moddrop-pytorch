import torch
import os
from PIL import Image
from torch.utils import data
# from torchvision import transforms
import pandas as pd
import numpy
from datasetVideoClassifier import DatasetVideoClassifier
from datasetVideoFeatureExtractor import DatasetVideoFeatureExtractor
from datasetSkeleton import DatasetSkeleton
from datasetAudio import DatasetAudio

import random
from datasetBasic import DatasetBasic
import re
import pickle
import glob
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

class DatasetOfDamagedMultimodal(data.Dataset):

    def __init__(self, root):
        """
        :type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
        """
        search_line = "*"

        self.file_list = glob.glob(root + search_line)
        self.dataset = []

        n = len(self.file_list)
        pct = n // 100
        i = 0
        for filepath in self.file_list:
            if i % pct == 0:
                print(f'i: {i}, {i/n * 100}% files found.')
            i += 1
            # # 先少弄点试试看
            # if i > 10000:
            #     break

            # with open(filepath, 'rb') as f:
            #     sample = pickle.load(f, encoding='iso-8859-1')
            #     # 读取的值默认是double型，这里改成float型，否则后面送进模型时会出错。
            #     for mdlt in sample['data'].keys():
            #         sample['data'][mdlt] = sample['data'][mdlt].astype(numpy.float32)
            #
            #     self.dataset.append( sample )

    def __getitem__(self, ind):
        """


        :param index:
        :return:
        """
        filepath = self.file_list[ind]
        with open(filepath, 'rb') as f:
            sample = pickle.load(f, encoding='iso-8859-1')
            # 读取的值默认是double型，这里改成float型，否则后面送进模型时会出错。
            for mdlt in sample['data'].keys():
                sample['data'][mdlt] = sample['data'][mdlt].astype(numpy.float32)
        # _s, label, QoU
        return sample['data'], sample['label'], sample['QoU']

    def __len__(self):
        return len(self.file_list)
