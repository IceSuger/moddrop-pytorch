import torch as t
import os
from PIL import Image
from torch.utils import data
# from torchvision import transforms
import pandas as pd
import numpy
import random
from datasetBasic import DatasetBasic
import re
import pickle
import glob
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

class DatasetSkeleton(DatasetBasic):

    def __init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes):
        """
        :type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
        """
        search_line = "*_g%02d*.pickle"
        filter_folder = 'filters/'
        self.stats_file = filter_folder + 'skeleton_stats'
        self._load_stats()


        modality_list = ['mocap']
        hand_list['both'] = modality_list
        DatasetBasic.__init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes, modality_list, search_line, block_size=36)


    # def prenormalize(self, x):
    #     x = x - numpy.mean(x)
    #     xstd = numpy.std(x)
    #     return x / (xstd + 0.00001)


    def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):
        """
        取一个时间块。即取一个5帧的块。stblock，意为space time block
        :param data_input:
        :param hnd:
        :param mdlt:
        :param start_frame:
        :return:
        """
        if start_frame is None:
            start_frame = random.randint(0, data_input['min_length']
                                         - self.step * (self.nframes - 1) - 1)
        stblock = data_input[hnd][mdlt][start_frame: start_frame
                                                     + self.step * (self.nframes - 1) + 1: self.step]
        return stblock, True

        #
        # goodness = False
        # if start_frame is None:
        #     start_frame = random.randint(0, len(data_input['min_length']) - self.step * (self.nframes - 1) - 1)
        # stblock = numpy.zeros([self.nframes, self.block_size, self.block_size])
        # for ii in range(self.nframes):
        #     v = data_input[hnd][mdlt][start_frame + ii * self.step]
        #     mm = abs(numpy.ma.maximum(v))
        #     if mm > 0.:
        #         # normalize to zero mean, unit variance,
        #         # concatenate in spatio-temporal blocks
        #
        #         stblock[ii] = self.prenormalize(v)    # [Xiao] [Debug] [2018-7-24 19:50]
        #         # stblock[ii] = v
        #         goodness = True
        # return stblock, goodness

    def _load_file(self, file_name, data_sample=None):
        # if data_sample is None:
        #     data_sample = {}
        # for hnd in self.hand_list:
        #     data_sample[hnd] = {}
        #     for mdlt in self.modality_list:
        #         if not hnd == 'both':
        #             for ind in ['a', 'l', 'r']:
        #                 file_name = re.sub('_' + ind + '_', '_' + hnd[0] + '_', file_name)
        #         for mdl in ['color', 'depth', 'mocap', 'descr', 'audio']:
        #             file_name = re.sub(mdl, mdlt, file_name)
        #         with open(file_name, 'rb') as f:
        #             [data_sample[hnd][mdlt]] = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        #             # print([data_sample[hnd][mdlt]])
        #         if not 'min_length' in data_sample:     # min_length 是当前这个样本中，最短长度模态的帧数
        #             data_sample['min_length'] = len(data_sample[hnd][mdlt])
        #         else:
        #             data_sample['min_length'] = min(data_sample['min_length'], len(data_sample[hnd][mdlt]))
        # return data_sample
        for mdlt in ['/depth/', '/color/', '/audio/']:
            file_name = re.sub(mdlt, '/mocap/', file_name)

        for hnd in ['_r_', '_l_']:
            file_name = re.sub(hnd, '_a_', file_name)
        for suff in ['depth', 'color', 'audio']:
            file_name = re.sub(suff, 'descr', file_name)

        if data_sample is None: data_sample = {}

        for hnd in self.hand_list:
            if not hnd in data_sample:
                data_sample[hnd] = {}
            for mdlt in self.modality_list:
                with open(file_name, 'rb') as f:
                    [d0, d1] = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
                data_sample[hnd][mdlt] = self._create_descriptor(d0, d1)
        if not 'min_length' in data_sample:
            data_sample['min_length'] = len(data_sample[hnd][mdlt])
        else:
            data_sample['min_length'] = min(data_sample['min_length'],
                                            len(data_sample[hnd][mdlt]))
        return data_sample

    def _create_descriptor(self, dscr0, dscr1):
        """ FOR PICKLES ONLY """
        dscr0 = self._normalize_descriptor0(dscr0)
        dscr1 = self._normalize_descriptor1(dscr1)
        return numpy.concatenate((dscr0, dscr1), axis=1)

    def _normalize_descriptor1(self, dscr):
        """ FOR PICKLES ONLY """
        dscr = numpy.subtract(dscr, self.mdt_mean2)
        return numpy.divide(dscr, numpy.sqrt(self.vdt_mean2))

    def _normalize_descriptor0(self, dscr):
        """ FOR PICKLES ONLY """
        dscr = numpy.subtract(dscr, self.mdt_mean1)
        return numpy.divide(dscr, numpy.sqrt(self.vdt_mean1))

    def _load_stats(self):
        """ FOR PICKLES ONLY """
        f = open(self.stats_file, 'rb')
        # self.mdt_mean1 = pickle.load(f)
        # self.mdt_mean2 = pickle.load(f)
        # self.vdt_mean1 = pickle.load(f)
        # self.vdt_mean2 = pickle.load(f)

        self.mdt_mean1 = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        self.mdt_mean2 = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        self.vdt_mean1 = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        self.vdt_mean2 = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        f.close()

        # # Theano variables
        # tensor4 = T.TensorType(theano.config.floatX, (False,) * 4)
        # self.sinputs = [T.tensor4('x')]


    # def _get_data_list(self, subset):
    #     """
    #     重写了父类的该方法
    #
    #     :type subset: string
    #     :param subset: string representing 'train', 'validation' or 'test' subsets
    #     """
    #     # self.data_list = data_list
    #     # self.search_line = search_line
    #
    #     if subset == 'train':
    #         folder = self.train_folder
    #     elif subset == 'valid':
    #         folder = self.valid_folder
    #     elif subset == 'test':
    #         folder = self.test_folder
    #     else:
    #         print('Unknown subset')
    #
    #     self.data_list[subset] = {}
    #     for cl in range(self.nclasses):
    #         list_right = glob.glob(folder + "*r%02d*.pickle" % (cl))
    #         list_left = glob.glob(folder + "*l%02d.pickle" % (cl))
    #         self.data_list[subset][cl] = list_right + list_left
