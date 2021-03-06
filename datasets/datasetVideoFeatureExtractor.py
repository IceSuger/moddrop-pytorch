# from torchvision import transforms
import numpy
import random
from datasets.datasetBasic import DatasetBasic
import re
import pickle
import glob
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

class DatasetVideoFeatureExtractor(DatasetBasic):

    def __init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes):
        """
        :type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
        """
        modality_list = ['color', 'depth']
        hand_list['both'] = modality_list
        DatasetBasic.__init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes, modality_list, block_size=36)
        # self.subset = subset
        # self.hand_list = hand_list
        # self.seq_per_class = seq_per_class
        # self.nclasses = nclasses
        # self.input_size = input_size
        # self.step = step
        # self.nframes = nframes
        #
        # # self.block_size = 36
        #
        # # self.dataset = {}
        # # self.dataset['train'] = {}
        # # self.dataset['valid'] = {}
        # # self.dataset['test'] = {}
        # # self.data_list = {}
        #
        # # Paths
        # self.search_line = "*_g%02d*.pickle"
        # self.input_folder = input_folder
        # self.train_folder = self.input_folder + modality + '/train/'
        # self.valid_folder = self.input_folder + modality + '/valid/'
        # self.test_folder = self.input_folder + modality + '/test/'
        #
        # self.modality_list = ['color', 'depth']
        # self.hand_list['both'] = self.modality_list
        # self.number_of_classes = 21


    def prenormalize(self, x):
        x = x - numpy.mean(x)
        xstd = numpy.std(x)
        return x / (xstd + 0.00001)


    def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):
        """
        取一个时间块。即取一个5帧的块。stblock，意为space time block
        :param data_input:
        :param hnd:
        :param mdlt:
        :param start_frame:
        :return:
        """
        goodness = False
        if start_frame is None:
            start_frame = random.randint(0, len(data_input['min_length']) - self.step * (self.nframes - 1) - 1)
        stblock = numpy.zeros([self.nframes, self.block_size, self.block_size])
        for ii in range(self.nframes):
            v = data_input[hnd][mdlt][start_frame + ii * self.step]
            mm = abs(numpy.ma.maximum(v))
            if mm > 0.:
                # normalize to zero mean, unit variance,
                # concatenate in spatio-temporal blocks

                stblock[ii] = self.prenormalize(v)    # [Xiao] [Debug] [2018-7-24 19:50]
                # stblock[ii] = v
                goodness = True
        return stblock, goodness

    def _load_file(self, file_name, data_sample=None):
        if data_sample is None:
            data_sample = {}
        for hnd in self.hand_list:
            data_sample[hnd] = {}
            for mdlt in self.modality_list:
                if not hnd == 'both':
                    for ind in ['a', 'l', 'r']:
                        file_name = re.sub('_' + ind + '_', '_' + hnd[0] + '_', file_name)
                for mdl in ['color', 'depth', 'mocap', 'descr', 'audio']:
                    file_name = re.sub(mdl, mdlt, file_name)

                if not os.path.isfile(file_name):  # 若文件不存在，就返回 None
                    return None

                with open(file_name, 'rb') as f:
                    [data_sample[hnd][mdlt]] = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
                    # print([data_sample[hnd][mdlt]])
                if not 'min_length' in data_sample:     # min_length 是当前这个样本中，最短长度模态的帧数
                    data_sample['min_length'] = len(data_sample[hnd][mdlt])
                else:
                    data_sample['min_length'] = min(data_sample['min_length'], len(data_sample[hnd][mdlt]))
        return data_sample


    def _get_data_list(self, subset):
        """
        重写了父类的该方法

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
        for cl in range(self.nclasses):
            list_right = glob.glob(folder + "*r%02d*.pickle" % (cl))
            list_left = glob.glob(folder + "*l%02d.pickle" % (cl))
            self.data_list[subset][cl] = list_right + list_left
