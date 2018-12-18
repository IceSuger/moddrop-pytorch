# from torchvision import transforms
import random
from datasets.datasetBasic import DatasetBasic
import re
import pickle
import glob
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

class DatasetAudio(DatasetBasic):

    def __init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes):
        """
        :type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
        """
        search_line = "*_g%02d*.pickle"

        hand_list = {}
        modality_list = ['audio']
        hand_list['both'] = modality_list
        DatasetBasic.__init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes, modality_list, search_line, block_size=36)


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
            start_frame = random.randint(0, data_input['min_length'] - self.step * (self.nframes - 1) - 1)
        end_ind = (start_frame + self.step * (self.nframes - 1) + 1) * 2
        stblock = data_input[hnd][mdlt][start_frame * 2: end_ind: self.step] * 20.
        return stblock, True


    def _load_file(self, file_name, data_sample=None):

        # for mdlt in ['/depth/', '/color/', '/audio/']:
        #     file_name = re.sub(mdlt, '/mocap/', file_name)
        #
        # for hnd in ['_r_', '_l_']:
        #     file_name = re.sub(hnd, '_a_', file_name)
        # for suff in ['depth', 'color', 'audio']:
        #     file_name = re.sub(suff, 'descr', file_name)
        #
        # if data_sample is None: data_sample = {}
        #
        # for hnd in self.hand_list:
        #     if not hnd in data_sample:
        #         data_sample[hnd] = {}
        #     for mdlt in self.modality_list:
        #         with open(file_name, 'rb') as f:
        #             [d0, d1] = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        #         data_sample[hnd][mdlt] = self._create_descriptor(d0, d1)
        # if not 'min_length' in data_sample:
        #     data_sample['min_length'] = len(data_sample[hnd][mdlt])
        # else:
        #     data_sample['min_length'] = min(data_sample['min_length'],
        #                                     len(data_sample[hnd][mdlt]))
        # return data_sample
        for mdlt in ['/depth/', '/color/', '/mocap/']:
            audio_dir = '/audio/'
            # if ("realtest" in file_name):
            #     audio_dir = '/audio/'
            # else:
            #     audio_dir = '/audio_cleaned/'
            file_name = re.sub(mdlt, audio_dir, file_name)  # edit to rename 'audio_cleaned' when training
        for hnd in ['_r_', '_l_']:
            file_name = re.sub(hnd, '_a_', file_name)
        for suff in ['depth', 'color', 'descr']:
            file_name = re.sub(suff, 'audio', file_name)
        # if not os.path.isfile(file_name):
        #     # print file_name
        #     file_name = glob.glob(file_name[:-14] + '*')[0]

        if not os.path.isfile(file_name):   # 若文件不存在，就返回 None
            # print(f'NOT_EXIST = {file_name}')
            return None

        if data_sample is None:
            data_sample = {}
        for hnd in self.hand_list:
            if not hnd in data_sample:
                data_sample[hnd] = {}
            for mdlt in self.modality_list:
                with open(file_name, 'rb') as f:
                    [data_sample[hnd][mdlt]] = pickle.load(f, encoding='iso-8859-1')     # ！！这个编码指定！！很重要！！
        if not 'min_length' in data_sample:
            data_sample['min_length'] = len(data_sample[hnd][mdlt]) // 2
        else:
            data_sample['min_length'] = min(data_sample['min_length'], len(data_sample[hnd][mdlt]) // 2)
        return data_sample

