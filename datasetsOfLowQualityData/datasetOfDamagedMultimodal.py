from torch.utils import data
# from torchvision import transforms
import numpy

import pickle
import glob
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

class DatasetOfDamagedMultimodal(data.Dataset):

    def __init__(self, root, train_valid_test='train', phi_s=None, QoU2delta_df=None):
        """
        :type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
        """
        search_line = "*"

        self.input_folder = root
        # self.train_folder = self.input_folder + 'train/'
        # self.valid_folder = self.input_folder + 'valid/'

        self.file_path = self.input_folder + train_valid_test + '/'
        self.file_list = glob.glob(self.file_path + search_line)
        self.dataset = []

        # n = len(self.file_list)
        # pct = n // 100
        # i = 0
        # for filepath in self.file_list:
        #     if i % pct == 0:
        #         print(f'i: {i}, {i/n * 100}% files found.')
        #     i += 1
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

    def _my_getitem__(self, ind):
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

    def __getitem__(self, ind):
        data, label, QoU = self._my_getitem__(ind)
        # print(f'label = {label}, type(label) = {type(label)}')
        return data, label[0]

    def __len__(self):
        return len(self.file_list)
