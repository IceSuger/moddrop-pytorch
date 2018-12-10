import torch
import os
from PIL import Image
from torch.utils import data
# from torchvision import transforms
import pandas as pd
import numpy
import random
import glob
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

class DatasetBasic(data.Dataset):

    def __init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes, modality_list, search_line='*_g%02d*.pickle', block_size=36):
        """
        :type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
        """
        self.subset = subset
        self.hand_list = hand_list

        self.seq_per_class = seq_per_class
        # if subset == 'train':
        #     self.seq_per_class = seq_per_class
        # else:   # valid or test
        #     self.seq_per_class = seq_per_class // 10

        self.nclasses = nclasses
        self.input_size = input_size
        self.step = step
        self.nframes = nframes
        self.modality_list = modality_list
        self.block_size = block_size

        self.dataset = {}
        self.dataset['train'] = {}
        self.dataset['valid'] = {}
        self.dataset['test'] = {}
        self.data_list = {}

        # Paths
        self.search_line = search_line
        self.input_folder = input_folder
        self.train_folder = self.input_folder + modality + '/train/'
        self.valid_folder = self.input_folder + modality + '/valid/'
        self.test_folder = self.input_folder + modality + '/test/'


        # Allocate memory for each modality
        for hnd in self.hand_list:
            self.dataset[subset][hnd] = {}
            for mdlt in self.hand_list[hnd]:
                self.dataset[subset][hnd][mdlt] = numpy.zeros([self.seq_per_class \
                                                               * self.nclasses] + self.input_size[mdlt])
                print(f'self.dataset[{subset}][{hnd}][{mdlt}] size is : {self.dataset[subset][hnd][mdlt].shape}')
        self.dataset[subset]['labels'] = numpy.zeros((self.seq_per_class * self.nclasses,))
        print(f"self.dataset[{subset}]['labels'] size is : {self.dataset[subset]['labels'].shape}")

        # [Xiao] [Debug]
        print('dataset_basic, __init__')
        # print(self.dataset)


        sample = 0
        class_number = 0

        self._get_data_list('valid')
        # self._load_dataset('valid')
        self._get_data_list('train')

        # print(f'self.data_list[{subset}][{class_number}] is : {self.data_list[subset][class_number]}')

        # Loading the data
        while sample < self.nclasses * self.seq_per_class:
            # Loading random gesture from a given class
            file_number = random.randint(0, len(self.data_list[subset][class_number]) - 1)
            file_name = self.data_list[subset][class_number][file_number]
            data_sample = self._load_file(file_name)
            # print(data_sample)

            # Extract a random spatio-temporal block from the file
            if data_sample['min_length'] >= self.step * (self.nframes - 1) + 1:     # 满足这个条件，说明该帧序列的长度，够取一个5帧的块的
                seq_number = random.randint(0, data_sample['min_length'] -
                                            self.step * (self.nframes - 1) - 1)
                ifloaded = False
                # print(f'self.hand_list keys: {self.hand_list.keys()}')
                for hnd in self.hand_list:  # ['both']
                    # print(f'self.hand_list[{hnd}] : {self.hand_list[hnd]}')
                    for mdlt in self.hand_list[hnd]:    # ['color', 'depth']
                        self.dataset[subset][hnd][mdlt][sample], ifl = \
                            self._get_stblock(data_sample, hnd, mdlt, seq_number)   # self.dataset[subset][hnd][mdlt] 是一个numpy array，见上面numpy.zeros那块初始化
                        ifloaded = ifl | ifloaded
                # If the block is loaded, proceed to the next class
                if ifloaded:
                    self.dataset[subset]['labels'][sample] = class_number   # int
                    sample += 1
                    class_number += 1
                    if class_number == self.nclasses:
                        class_number = 0

        # Reshape the data, convert to floatX
        for hnd in self.hand_list:
            for mdlt in self.hand_list[hnd]:
                if self.input_size[mdlt][0] == self.nframes:
                    self.dataset[subset][hnd][mdlt] = \
                        self.dataset[subset][hnd][mdlt].reshape([self.seq_per_class \
                                                                 * self.nclasses, self.input_size[mdlt][0], 1] \
                                                                + self.input_size[mdlt][1:]).astype(
                            numpy.float32)
                else:
                    self.dataset[subset][hnd][mdlt] = \
                        self.dataset[subset][hnd][mdlt].reshape([self.seq_per_class \
                                                                 * self.nclasses, 1] \
                                                                + self.input_size[mdlt]).astype(numpy.float32)

        # self.dataset[subset]['labels'] = numpy.int8(self.dataset[subset]['labels'])

        # [Xiao] [Debug]
        print('dataset_basic, after __init__')
        # print(self.dataset)


    def __getitem__(self, ind):
        """


        :param index:
        :return:
        """
        subset = self.subset

        inputs = []
        label = -1

        # [Xiao] [Debug]
        # print(self.dataset)

        # print(f'hand_list keys: {self.hand_list.keys()}')
        # print(f"hand_list['left'] : {self.hand_list['left']}")
        # print(f"hand_list['both'] : {self.hand_list['both']}")
        # Append data from all channels

        for hnd in self.hand_list:
            for mdlt in self.hand_list[hnd]:
                # print(f'hnd: {hnd}, mdlt: {mdlt}, ind: {ind}')
                inputs.append(self.dataset[subset][hnd][mdlt][ind])
                # print(f'inputs length: {len(inputs)}')
                # print(f'inputs item type: {type(inputs[0])}')
                # print(f'inputs item size: {inputs[0].shape}')
        if subset in ['train', 'valid']:
            # label.append(self.dataset[subset]['labels'][ind])
            label = self.dataset[subset]['labels'][ind]
            # print(label)
        # return inputs, label
        # print(f'In datasetBasic.py, inputs = {inputs}')
        # print(f'inputs[0].shape = {inputs[0].shape}')
        # print(f'len(inputs) = {len(inputs)}')
        # print(f'type(inputs) = {type(inputs)}')
        return torch.tensor(inputs), int(label)
        # [以前是这样的，现在不是了。]this is a list of tuples of size (batch_size, channel=1, input_size)

    def __len__(self):
        # return 42
        return self.seq_per_class * self.nclasses

    def _get_data_list(self, subset):
        """
        Function to retrieve list of training/validating/testing data filenames.

        :type subset: string
        :param subset: string representing 'train', 'validation' or 'test' subsets
        """
        # self.data_list = data_list
        # self.search_line = search_line

        if subset == 'train':
            folder = self.train_folder
        elif subset == 'valid':
            folder = self.valid_folder
        elif subset == 'test':
            folder = self.test_folder
        else:
            print('Unknown subset')

        self.data_list[subset] = {}
        # print(f'In _get_data_list, folder is {folder}, search_line is: {self.search_line}')
        for cl in range(self.nclasses):
            self.data_list[subset][cl] = glob.glob(folder + self.search_line % (cl))

        # print('_get_date_list, subset = train')
        # print(self.data_list['train'].keys())
