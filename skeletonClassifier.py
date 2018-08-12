import torch
import numpy
import random
from torch import nn
import re
import pickle
import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

from basicClassifier import basicClassifier

class SkeletonExtractorNet(nn.Module):
    def __init__(self, num_of_classes, dlength):
        """

        :param num_of_classes:
        :param dlength: length of pose-descriptor
        """
        super().__init__()
        self.num_of_classes = num_of_classes
        self.dlength = dlength
        self.nframes = 5

        #self.input = shape=(self.batch_size, 1, self.block_size, self.block_size, self.nframes)
        # shape = (batch_size, nframes, 1, dlength)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=.0),
            nn.Linear(self.dlength * self.nframes, 700),
            nn.Tanh(),
            nn.BatchNorm1d(700),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=.0),
            nn.Linear(700, 400),
            nn.Tanh(),
            nn.BatchNorm1d(400),
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(p=.0),
            nn.Linear(400, 350),
            nn.Tanh(),
            nn.BatchNorm1d(350),
        )

        self.output_block = nn.Sequential(
            nn.Dropout(p = .2),
            nn.Linear(350, self.num_of_classes),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(self.num_of_classes)
        )



    def forward(self, x):
        # print(f'In Skeleton, x size is: {x.shape}')
        x = x.squeeze()
        # print(f'In Skeleton, after squeeze(), x size is: {x.shape}')

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_block(x)

        # x = x.permute(1, 0, 2, 3, 4, 5)
        # # print(f'x size_after_permutation is: {x.shape}')
        # # # x = x.squeeze()
        # # print(f'x[0] size is: {x[0].shape}')
        #
        # x0 = x[0].permute(0, 2, 1, 3, 4) # 维度换位, 换完是
        # x1 = x[1].permute(0, 2, 1, 3, 4)
        # # print('after permutation')
        # # print(x.shape)
        # # print('')
        # x0 = self.block1_color(x0)
        # x1 = self.block1_depth(x1)
        # # print('after 1st block')
        # # print(x.shape)
        # # print('')
        # # # print的结果为：torch.Size([32, 25, 1, 158, 88])
        # # x = x.view(x.size(0), x.size(1), x.size(2), -1)
        # x0 = x0.squeeze()
        # x1 = x1.squeeze()
        # # print('after 1st reshape')
        # # print(x.shape)
        # # print('')
        # # # print的结果为：torch.Size([32, 25, 158, 88])
        # x0 = self.block2_color(x0)
        # x1 = self.block2_depth(x1)
        # # print('after 2nd block')
        # # print(x.shape)
        # # print('')
        # # # print的结果为：torch.Size([32, 25, 154, 84])
        # x0 = x0.view(x0.size(0), -1)
        # x1 = x1.view(x1.size(0), -1)
        # # print('after 2nd flatten')
        # # print(x.shape)
        # # print('')
        # # # print的结果为：torch.Size([32, 323400])
        # # print(f'x0 size is: {x0.shape}')
        # # print(f'x1 size is: {x1.shape}')
        # x = torch.cat([x0, x1], 1)
        # # print(f'x size after concat, and before reshape: {x.shape}')
        # # # x = x.view(1, -1)
        # # print(f'x shape before fusion block: {x.shape}')
        # x = self.block_fusion(x)
        # # print(f'x shape after fusion block: {x.shape}, And x is : {x}')
        # x = self.output_block(x)

        return x

class skeletonClassifier(basicClassifier):
    def __init__(self, input_folder, filter_folder, number_of_classes=21,
                 step=4, nframes=5, block_size=36, batch_size=42, pretrained=False):
        basicClassifier.__init__(self, input_folder, filter_folder, number_of_classes,
									 step, nframes, batch_size, 'mocap', pretrained)


        self.dlength = 183

        self.block_size = block_size    # size of a bounding box surrouding each hand

        self.input_size['mocap'] = [self.nframes, self.dlength]
        # self.modality_list = ['mocap']
        # self.hand_list['both'] = self.modality_list


        self.number_of_classes = 21

        # Paths
        self.filters_file = filter_folder + 'skeletonClassifier_step' + str(step) + '.npz'

        # Training parameters
        self.learning_rate_value = 0.2
        self.learning_rate_decay = 0.9995
        self.n_epochs = 5000

        # [Xiao] 用于保存模型
        self.model_name = 'skeletonClassifier'



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
        self.network = SkeletonExtractorNet(self.number_of_classes, self.dlength)
        self.model = self.network
        return self.network






