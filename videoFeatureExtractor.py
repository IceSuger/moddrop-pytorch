import torch
import numpy
import random
from torch import nn
import re
import pickle
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

from basicClassifier import basicClassifier

class VideoFeatureExtractorNet(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()
        self.num_of_classes = num_of_classes

        #self.input = shape=(self.batch_size, 1, self.block_size, self.block_size, self.nframes)

        # self.conv_layers = [(25, 3, 1, 5, 5), (25, 25, 5, 5)]
        self.block1_color = nn.Sequential(
            # sample size: channel = 5, height = 36, width = 36
            # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv3d(1, 25, kernel_size=(3,5,5), stride=1),
            nn.BatchNorm3d(25),
            nn.MaxPool3d((3,2,2))
        )

        self.block2_color = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(25, 25, kernel_size=(5,5)),
            nn.BatchNorm2d(25),
            nn.MaxPool2d((1,1))
        )

        self.block_fusion = nn.Sequential(
            nn.Dropout(p=.0),
            nn.Linear(7200, 900),
            nn.Tanh(),
            nn.BatchNorm1d(900),

            nn.Dropout(p=.0),
            nn.Linear(900, 450),
            nn.Tanh(),
            nn.BatchNorm1d(450),

        )

        self.output_block = nn.Sequential(
            nn.Dropout(.0),
            nn.Linear(450, self.num_of_classes),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(self.num_of_classes)
        )

        # 开始 depth 相关的层
        self.block1_depth = nn.Sequential(
            # sample size: channel = 5, height = 36, width = 36
            # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv3d(1, 25, kernel_size=(3, 5, 5)),
            nn.BatchNorm3d(25),
            nn.MaxPool3d((3, 2, 2))
        )

        self.block2_depth = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(25, 25, kernel_size=(5, 5)),
            nn.BatchNorm2d(25),
            nn.MaxPool2d((1, 1))
        )

        # self.fc_block = nn.Sequential(
        #     nn.Dropout(p=.0),
        #     nn.Linear(323400,900),
        #     nn.Tanh(),
        #     nn.BatchNorm1d(900),
        #
        #     nn.Dropout(p=.0),
        #     nn.Linear(900, 450),
        #     nn.Tanh(),
        #     nn.BatchNorm1d(450),
        #
        #     nn.Dropout(.0),
        #     nn.Linear(450, self.num_of_classes),
        #     nn.Softmax(dim=1),
        #     nn.BatchNorm1d(self.num_of_classes)
        # )

    def forward(self, x):
        '''
        :param x: x[0] 为color的输入，x[1] 为 depth 的输入
        :return:
        '''
        # # 输入x的size为：torch.Size([32, 5, 1, 320, 180]) ，其中32为batch_size；5为连续5帧构成一个volume，故视为一个输入为5个通道；1是一个输入；320,180为高，宽
        print(f'In VideoFeat, x size is: {x.shape}')
        x = x.permute(1, 0, 2, 3, 4, 5)
        # print(f'x size_after_permutation is: {x.shape}')
        # # x = x.squeeze()
        # print(f'x[0] size is: {x[0].shape}')

        x0 = x[0].permute(0, 2, 1, 3, 4) # 维度换位, 换完是
        x1 = x[1].permute(0, 2, 1, 3, 4)
        # print('after permutation')
        # print(x.shape)
        # print('')
        x0 = self.block1_color(x0)
        x1 = self.block1_depth(x1)
        # print('after 1st block')
        # print(x.shape)
        # print('')
        # # print的结果为：torch.Size([32, 25, 1, 158, 88])
        # x = x.view(x.size(0), x.size(1), x.size(2), -1)
        x0 = x0.squeeze()
        x1 = x1.squeeze()
        # print('after 1st reshape')
        # print(x.shape)
        # print('')
        # # print的结果为：torch.Size([32, 25, 158, 88])
        x0 = self.block2_color(x0)
        x1 = self.block2_depth(x1)
        # print('after 2nd block')
        # print(x.shape)
        # print('')
        # # print的结果为：torch.Size([32, 25, 154, 84])
        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        # print('after 2nd flatten')
        # print(x.shape)
        # print('')
        # # print的结果为：torch.Size([32, 323400])
        # print(f'x0 size is: {x0.shape}')
        # print(f'x1 size is: {x1.shape}')
        x = torch.cat([x0, x1], 1)
        # print(f'x size after concat, and before reshape: {x.shape}')
        # # x = x.view(1, -1)
        # print(f'x shape before fusion block: {x.shape}')
        x = self.block_fusion(x)
        # print(f'x shape after fusion block: {x.shape}, And x is : {x}')
        x = self.output_block(x)

        return x

class videoFeatureExtractor(basicClassifier):
    def __init__(self, input_folder, filter_folder, number_of_classes=21,
                 step=4, nframes=5, block_size=36, batch_size=42, pretrained=False):
        basicClassifier.__init__(self, input_folder, filter_folder, number_of_classes,
									 step, nframes, batch_size, 'color', pretrained)

        self.block_size = block_size    # size of a bounding box surrouding each hand
        self.input_size['color'] = [self.nframes, self.block_size, self.block_size]
        self.input_size['depth'] = self.input_size['color']
        self.conv_layers = [(25, 3, 1, 5, 5), (25, 25, 5, 5)]
        self.pooling = [(2, 2, 3), (1, 1)]
        self.fc_layers = [900, 450, self.nclasses]
        self.dropout_rates = [0., 0., 0.]  # dropout rates for fully connected layers
        self.activations = [self.activation] * (len(self.conv_layers)
                                                + len(self.fc_layers) - 1)

        # self.modality_list = ['color', 'depth']
        # self.hand_list['both'] = self.modality_list
        self.number_of_classes = 21

        # lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results
        #
        # # Theano inputs
        #
        # tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
        # self.sinputs = [tensor5(mdlt) for mdlt in self.modality_list]
        # self.network = {}

        # Paths
        self.filters_file = filter_folder + 'videoFeatureExtractor_step' + str(step) + '.npz'

        # Training parameters
        self.learning_rate_value = 0.01
        self.learning_rate_decay = 0.9998
        self.n_epochs = 5000

        # [Xiao] 用于保存模型
        self.model_name = 'videoFeatureExtractor'



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
        self.network = VideoFeatureExtractorNet(self.number_of_classes)
        self.model = self.network
        return self.network






