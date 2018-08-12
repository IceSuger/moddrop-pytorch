import torch
import numpy
import random
from torch import nn
import re
import pickle
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

from basicClassifier import basicClassifier

class AudioExtractorNet(nn.Module):
    def __init__(self, num_of_classes):
        """

        :param num_of_classes:
        """
        super().__init__()
        self.num_of_classes = num_of_classes
        input_size = [9, 40]
        kernel_size = (5,5)
        conv_stride = 1
        conv_padding = 0

        self.conv2d_1 = nn.Sequential(
            # nn.Dropout(p=.0),
            nn.Conv2d(1, 25, kernel_size, conv_stride),
            nn.Tanh(),
            nn.BatchNorm2d(25),
        )
        torch.nn.init.xavier_uniform(self.conv2d_1[0].weight)

        self.maxPool2d_1 = nn.MaxPool2d((1,1))

        # 卷积之后的图片尺寸，以宽为例（高类似）： W1 = （W0 - K + 2P） / S + 1
        # 其中，W1、W0为卷积前后的图片宽度，K为卷积核宽度，P为padding尺寸，S为stride大小
        newWidth = (input_size[0] - kernel_size[0] + 2 * conv_padding) / conv_stride + 1
        newHeight = (input_size[1] - kernel_size[1] + 2 * conv_padding) / conv_stride + 1
        linear_in_size = newWidth * newHeight * 25  # 25 为卷积核个数

        self.fc1 = nn.Sequential(
            nn.Dropout(p=.0),
            nn.Linear(linear_in_size, 700),
            nn.Tanh(),
            nn.BatchNorm1d(700),
        )

        self.fcn = nn.Sequential(
            nn.Dropout(p=.0),
            nn.Linear(700, 350),
            nn.Tanh(),
            nn.BatchNorm1d(350),
        )

        self.output_block = nn.Sequential(
            nn.Dropout(p = .2),
            nn.Linear(350, self.num_of_classes),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(self.num_of_classes)
        )

        print('audioClassifierNet is :')
        # print(self.state_dict())
        # print(self.named_modules())
        for idx, m in enumerate(self.named_modules()):
            print(idx, '->', m)



    def forward(self, x):
        # input size (42, 1, 1, 9, 40)
        # print(f'In Audio, x size is: {x.shape}')
        x = x.squeeze(1)
        # print(f'In Audio, after squeeze(), x size is: {x.shape}')

        # x = x.view(x.size(0), -1)
        x = self.conv2d_1(x)
        x = self.maxPool2d_1(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fcn(x)
        x = self.output_block(x)

        return x

class audioClassifier(basicClassifier):
    def __init__(self, input_folder, filter_folder, number_of_classes=21,
                 step=4, nframes=5, block_size=36, batch_size=42, pretrained=False):
        basicClassifier.__init__(self, input_folder, filter_folder, number_of_classes,
									 step, nframes, batch_size, 'audio', pretrained)


        self.dlength = 183

        self.block_size = block_size    # size of a bounding box surrouding each hand

        self.input_size['audio'] = [9, 40]


        self.number_of_classes = 21

        # Training parameters
        self.learning_rate_value = 0.05
        self.learning_rate_decay = 0.9999
        self.n_epochs = 5000

        # [Xiao] 用于保存模型
        self.model_name = 'audioClassifier'



    # !!!!some functions are not copyed here.

    # def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):
    #     goodness = False
    #     if start_frame is None:
    #         start_frame = random.randint(0, len(data_input['min_length']) - self.step * (self.nframes - 1) - 1)
    #     stblock = numpy.zeros([self.nframes, self.block_size, self.block_size])
    #     for ii in range(self.nframes):
    #         v = data_input[hnd][mdlt][start_frame + ii * self.step]
    #         mm = abs(numpy.ma.maximum(v))
    #         if mm > 0.:
    #             # normalize to zero mean, unit variance,
    #             # concatenate in spatio-temporal blocks
    #             stblock[ii] = self.prenormalize(v)
    #             goodness = True
    #     return stblock, goodness
    #
    # def _load_file(self, file_name, data_sample=None):
    #     if data_sample is None:
    #         data_sample = {}
    #     for hnd in self.hand_list:
    #         data_sample[hnd] = {}
    #         for mdlt in self.modality_list:
    #             if not hnd == 'both':
    #                 for ind in ['a', 'l', 'r']:
    #                     file_name = re.sub('_' + ind + '_', '_' + hnd[0] + '_', file_name)
    #             for mdl in ['color', 'depth', 'mocap', 'descr', 'audio']:
    #                 file_name = re.sub(mdl, mdlt, file_name)
    #             with open(file_name, 'rb') as f:
    #                 [data_sample[hnd][mdlt]] = pickle.load(f)
    #                 print([data_sample[hnd][mdlt]])
    #             if not 'min_length' in data_sample:
    #                 data_sample['min_length'] = len(data_sample[hnd][mdlt])
    #             else:
    #                 data_sample['min_length'] = min(data_sample['min_length'], len(data_sample[hnd][mdlt]))
    #     return data_sample

    # [Xiao]
    def build_network(self, input_var = None, batch_size = None):
        if not input_var is None:
            self.sinputs = input_var
        if not batch_size is None:
            self.batch_size = batch_size

        # 构建网络，在这里 new 一个 Net 对象
        self.network = AudioExtractorNet(self.number_of_classes)
        self.model = self.network
        return self.network






