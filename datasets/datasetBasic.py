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

from CONSTS import STEP_TO_NEXT_STBLOCK

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
        self.dataset['train'] = []
        self.dataset['valid'] = []
        self.dataset['test'] = []
        # self.dataset['train'] = {}
        # self.dataset['valid'] = {}
        # self.dataset['test'] = {}
        self.data_list = {}

        # Paths
        self.search_line = search_line
        self.input_folder = input_folder
        self.train_folder = self.input_folder + modality + '/train/'
        self.valid_folder = self.input_folder + modality + '/valid/'
        self.test_folder = self.input_folder + modality + '/test/'


        # # v2.6 不再需要先分配空间了
        # # Allocate memory for each modality
        # for hnd in self.hand_list:
        #     self.dataset[subset][hnd] = {}
        #     for mdlt in self.hand_list[hnd]:
        #         # self.dataset[subset][hnd][mdlt] = numpy.zeros([self.seq_per_class \
        #         #                                                * self.nclasses] + self.input_size[mdlt])
        #         self.dataset[subset][hnd][mdlt] = []
        #         # print(f'self.dataset[{subset}][{hnd}][{mdlt}] size is : {self.dataset[subset][hnd][mdlt].shape}')
        # # self.dataset[subset]['labels'] = numpy.zeros((self.seq_per_class * self.nclasses,))
        # self.dataset[subset]['labels'] = []
        # # print(f"self.dataset[{subset}]['labels'] size is : {self.dataset[subset]['labels'].shape}")

        # [Xiao] [Debug]
        print('dataset_basic, __init__')
        # print(self.dataset)

        self._get_data_list('valid')
        # self._load_dataset('valid')
        self._get_data_list('train')

        # print(f'self.data_list[{subset}][{class_number}] is : {self.data_list[subset][class_number]}')

        # Loading the data
        if seq_per_class >= 0:
            # [Xiao] v2.6.1 暂时用这个参数来表明，当前单模态数据及是属于多模态的(seq_per_class < 0)，还是单独自己干活的。如果是前者，就直接跳过 self._load_data 的过程，因为想要的主要是 _get_stblock 等等方法而已。
            # [Xiao] v2.6 去掉读数据这里的随机性！
            self._load_data(subset)

        print(f'dataset length = {self.__len__()}')

        # print(f'self.input_size = {self.input_size}')

        # # Reshape the data, convert to floatX
        # for hnd in self.hand_list:
        #     for mdlt in self.hand_list[hnd]:
        #         if self.input_size[mdlt][0] == self.nframes:
        #             # self.dataset[subset][hnd][mdlt] = \
        #             #     self.dataset[subset][hnd][mdlt].reshape([self.seq_per_class \
        #             #                                              * self.nclasses, self.input_size[mdlt][0], 1] \
        #             #                                             + self.input_size[mdlt][1:]).astype(
        #             #         numpy.float32)
        #             self.dataset[subset][hnd][mdlt] = numpy.array(self.dataset[subset][hnd][mdlt]).reshape([-1, self.input_size[mdlt][0], 1] \
        #                                                         + self.input_size[mdlt][1:]).astype(numpy.float32)
        #         else:
        #             # self.dataset[subset][hnd][mdlt] = \
        #             #     self.dataset[subset][hnd][mdlt].reshape([self.seq_per_class \
        #             #                                              * self.nclasses, 1] \
        #             #                                             + self.input_size[mdlt]).astype(numpy.float32)
        #             self.dataset[subset][hnd][mdlt] = numpy.array(self.dataset[subset][hnd][mdlt]).reshape([-1, 1] \
        #                                                         + self.input_size[mdlt]).astype(numpy.float32)
        #
        #             # self.dataset[subset]['labels'] = numpy.int8(self.dataset[subset]['labels'])
        #         print(f'self.dataset[{subset}][{hnd}][{mdlt}] shape = {self.dataset[subset][hnd][mdlt].shape}')

        # [Xiao] [Debug]
        print('dataset_basic, after __init__')
        # print(self.dataset)


    def __getitem__0(self, ind):
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

    def __getitem__(self, ind):
        # ind -> (file_name, start_frame, class_number)
        file_name, start_frame, label = self.dataset[self.subset][ind]

        inputs = self._get_data_by_filename_and_startframe(file_name, start_frame)

        return inputs, int(label)



    def __len__(self):
        # return 42
        # return self.seq_per_class * self.nclasses
        return len(self.dataset[self.subset])

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

    def _load_data0(self, subset):
        # 遍历class_number
        for class_number in range(self.nclasses):
            # 遍历当前 class 对应的 data_list
            for file_name in self.data_list[subset][class_number]:
                data_sample = self._load_file(file_name)

                # 遍历该文件内能提取的所有 stblock，每提取一个都作为一个样本存到 self.dataset 里
                block_origin_frame_cnt = self.step * (self.nframes - 1) + 1
                start_frame_range = data_sample['min_length'] - block_origin_frame_cnt + 1
                for seq_number in range(start_frame_range): # 得保证该原始帧序列的长度，够取一个5帧的块的（即，多于17帧）
                    # v2.6.0.1 如果遍历所有可能的stblock的话，总的训练样本数高达60万多，没必要。所以干脆对每个序列都只取第一帧开始的1个 stblock 吧。
                    if seq_number >= 1:
                        break

                    ifloaded = False

                    for hnd in self.hand_list:  # ['both']
                        for mdlt in self.hand_list[hnd]:  # ['color', 'depth']
                            tmp_data, ifl = self._get_stblock(data_sample, hnd, mdlt, seq_number)
                            self.dataset[subset][hnd][mdlt].append(tmp_data.copy())
                            ifloaded = ifl | ifloaded
                    # If the block is loaded, proceed to the next class
                    if ifloaded:
                        self.dataset[subset]['labels'].append(class_number)  # int

    def _load_data(self, subset):
        # 遍历class_number
        for class_number in range(self.nclasses):
            # 遍历当前 class 对应的 data_list
            n = len(self.data_list[self.subset][class_number])
            ten_pct = n // 10
            i = 0
            for file_name in self.data_list[self.subset][class_number]:
                if i % ten_pct == 0:
                    print(f'{self.__class__.__name__}, class_number = {class_number}, _load_data, {i / n}')
                i += 1

                # 当前的 file_name 只是其中某个模态的，需要依次访问其对应的各个模态的文件，并遍历所有可能的 start_frame，对每种合法的，都存入 self.dataset
                # legal = False
                start_frame = 0
                while True:
                    if_legal = self._check_if_can_form_stblock(file_name, start_frame)
                    if not if_legal:
                        break
                    # 能到这里，说明当前的 stblock 取法是合法的，则记到 self.dataset 中
                    self.dataset[self.subset].append((file_name, start_frame, class_number))

                    # 往后滑，继续尝试
                    start_frame += STEP_TO_NEXT_STBLOCK  # 滑动几帧，继续取下一个 stblock

    def _check_if_can_form_stblock(self, file_name, start_frame):
        data_sample = self._load_file(file_name)

        if data_sample is None:
            return False

        # 至此说明文件存在，则下一步就是看看当前模态下的这个文件里的数据够不够提取出一个 以 start_frame 起始的 stblock
        # 遍历该文件内能提取的所有 stblock，每提取一个都作为一个样本存到 self.dataset 里
        block_origin_frame_cnt = self.step * (self.nframes - 1) + 1     # step = 4 时，一个 5 帧的块，总长 17 帧
        start_frame_range = data_sample['min_length'] - block_origin_frame_cnt + 1

        # 如果 start_frame 超出了 start_frame_range，就不能提取出完整的 stblock 了
        if start_frame >= start_frame_range:
            return False

        # 顺利至此则说明可以提取合法的 stblock
        return True

    def _get_data_by_filename_and_startframe(self, file_name, start_frame):
        inputs = []

        for hnd in self.hand_list:
            for mdlt in self.hand_list[hnd]:
                data_sample = self._load_file(file_name)
                tmp_data, ifl = self._get_stblock(data_sample, hnd, mdlt, start_frame)

                inputs.append(tmp_data)

        if self.input_size[mdlt][0] == self.nframes:
            inputs = numpy.array(inputs).reshape([-1, self.input_size[mdlt][0], 1] + self.input_size[mdlt][1:]).astype(
                numpy.float32)
        else:
            inputs = numpy.array(inputs).reshape([-1, 1] + self.input_size[mdlt]).astype(numpy.float32)

        return torch.tensor(inputs)
