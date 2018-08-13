import torch
import numpy
import random
from torch import nn
import re
import pickle
import os
from collections import OrderedDict
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

from videoFeatureExtractor import videoFeatureExtractor
from videoFeatureExtractor import VideoFeatureExtractorNet


class VideoClassifierNet(nn.Module):
    def __init__(self, num_of_classes, extractorPath=None):
        super().__init__()
        self.num_of_classes = num_of_classes

        if extractorPath is None:
            print(f'Please set extractorPath.')
            return

        # 建立这俩网络
        self.left_network = VideoFeatureExtractorNet(num_of_classes)
        self.right_network = VideoFeatureExtractorNet(num_of_classes)

        # print('self.left_network.children() is :')
        # for idx, m in enumerate(self.left_network.children()):
        #     print(idx, '->', m)

        # 然后加载预训练的模型
        extractor_dict = torch.load(extractorPath)
        self.left_network.load_state_dict(extractor_dict)
        self.right_network.load_state_dict(extractor_dict)

        # print('======== after load weights ============')
        # print('self.left_network.children() is :')
        # for idx, m in enumerate(self.left_network.named_children()):
        #     print(idx, '->', m)

        # 然后去掉最后的层
        # extractor_dict = {k:v for k, v in extractor_dict.items() if not k in ['']}3
        # od = OrderedDict(*list(self.left_network.named_children()))
        self.left_network = nn.Sequential(OrderedDict(list(self.left_network.named_children())))
        self.right_network = nn.Sequential(OrderedDict(list(self.right_network.named_children())))
        # self.left_network = nn.Sequential(*list(self.left_network.named_children())[:-4])
        # self.right_network = nn.Sequential(*list(self.right_network.named_children())[:-4])

        # 加上新的融合层(先在 forward 中完成concat，即torch.cat)
        self.fc3 = nn.Sequential(
            nn.Dropout(p=.0),
            # todo
            nn.Linear(450*2, 84),
            nn.Tanh(),
            nn.BatchNorm1d(84),
        )

        self.output = nn.Sequential(
            nn.Dropout(.0),
            nn.Linear(84, self.num_of_classes),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(self.num_of_classes)
        )

        # print('videoClassifierNet is :')
        # # print(self.state_dict())
        # # print(self.named_modules())
        # for idx, m in enumerate(self.named_modules()):
        #     print(idx, '->', m)


    def forward(self, x):
        '''
        :param x: x[0] 为右手的输入，x[1] 为左手的输入
        :return:
        '''
        # # 输入x的size为：torch.Size([batchsize, handcnt, color&depth(即2), 5, 1, 320, 180]) ，其中32为batch_size；5为连续5帧构成一个volume，故视为一个输入为5个通道；1是一个输入；320,180为高，宽

        x = x.permute(1, 0, 2, 3, 4, 5)
        print(f'In video classifier, x size is: {x.shape}')
        x0 = x[:2].permute(1, 0, 2, 3, 4, 5)
        x1 = x[2:].permute(1, 0, 2, 3, 4, 5)

        print(f'In video classifier, after permutation, x0 size is: {x0.shape}')

        # x0 = self.right_network.forward(x0)
        # x1 = self.left_network.forward(x1)
        x0 = self.videoFeatForward(self.right_network, x0)
        x1 = self.videoFeatForward(self.left_network, x1)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)

        x = torch.cat([x0, x1], 1)

        x = self.fc3(x)
        x = self.output(x)

        return x

    def videoFeatForward(self, network, x):
        '''
        :param x: x[0] 为color的输入，x[1] 为 depth 的输入
        :return:
        '''
        # # 输入x的size为：torch.Size([32, 5, 1, 320, 180]) ，其中32为batch_size；5为连续5帧构成一个volume，故视为一个输入为5个通道；1是一个输入；320,180为高，宽
        # print(f'In VideoFeat, x size is: {x.shape}')
        x = x.permute(1, 0, 2, 3, 4, 5)
        # print(f'x size_after_permutation is: {x.shape}')
        # # x = x.squeeze()
        # print(f'x[0] size is: {x[0].shape}')

        x0 = x[0].permute(0, 2, 1, 3, 4) # 维度换位, 换完是
        x1 = x[1].permute(0, 2, 1, 3, 4)
        # print('after permutation')
        # print(x.shape)
        # print('')
        x0 = network.block1_color(x0)
        x1 = network.block1_depth(x1)
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
        x0 = network.block2_color(x0)
        x1 = network.block2_depth(x1)
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
        x = network.block_fusion(x)

        return x

class videoClassifier(videoFeatureExtractor):
    def __init__(self, input_folder, filter_folder, number_of_classes=21, step=4, nframes=5,
                 block_size=36, batch_size=42, use_standard_features=True, pretrained=False):
        videoFeatureExtractor.__init__(self, input_folder, filter_folder, number_of_classes,
                 step, nframes, block_size, batch_size, pretrained)

        # [Xiao] 用于保存模型
        self.model_name = 'videoClassifier'

        # [Xiao] 指定已经预训练好的网络参数保存路径
        self.extractorPath = 'checkpoints/videoFeatureExtractor.pth'



        # # symbolic inputs
        # self.hand_list = {}
        # for hnd in ['right', 'left']:
        #     self.hand_list[hnd] = self.modality_list

    def build_network(self, input_var = None, batch_size = None):
        if not input_var is None:
            self.sinputs = input_var
        if not batch_size is None:
            self.batch_size = batch_size

        # 构建网络，在这里 new 一个 Net 对象
        self.network = VideoClassifierNet(self.number_of_classes, self.extractorPath)
        self.model = self.network
        return self.network