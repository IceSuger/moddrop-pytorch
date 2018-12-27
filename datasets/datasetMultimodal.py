import glob

from torch.utils.data import Dataset

from CONSTS import STEP_TO_NEXT_STBLOCK, DEBUGGING
from datasets.datasetAudio import DatasetAudio
from datasets.datasetSkeleton import DatasetSkeleton
from datasets.datasetVideoClassifier import DatasetVideoClassifier


class DatasetMultimodal(Dataset):
    def __init__(self, input_folder, modality, subset, hand_list, seq_per_class, nclasses, input_size, step, nframes):

        print(f'{self.__class__.__name__} dataset')

        # Paths
        self.search_line = "*_g%02d*.pickle"
        self.input_folder = input_folder
        self.train_folder = self.input_folder + modality + '/train/'
        self.valid_folder = self.input_folder + modality + '/valid/'
        self.test_folder = self.input_folder + modality + '/test/'

        hand_list['right'] = ['color', 'depth']
        hand_list['left'] = ['color', 'depth']
        hand_list['both'] = ['mocap', 'audio']

        self.seq_per_class = seq_per_class # v2.6之后完全没用了
        self.step = step
        self.nframes = nframes
        self.nclasses = nclasses
        self.subset = subset

        self.dataset = {}
        self.dataset['train'] = []
        self.dataset['valid'] = []
        self.dataset['test'] = []
        self.data_list = {}
        # for mdlt in self.modality_list:
        #     self.dataset[self.subset][mdlt] = []
        # self.dataset[self.subset]['labels'] = []


        # 各模态数据集类
        self.datasetTypes = {
            'video': DatasetVideoClassifier(input_folder, 'color', subset, hand_list,
                                            -1,
                                            nclasses, input_size, step, nframes),
            'mocap': DatasetSkeleton(input_folder, 'mocap', subset, hand_list,
                                     -1,
                                     nclasses, input_size, step, nframes),
            'audio': DatasetAudio(input_folder, 'audio', subset, hand_list,
                                  -1,
                                  nclasses, input_size, step, nframes),
        }
        self.modality_list = list(self.datasetTypes.keys())

        # 获取文件名列表
        self._get_data_list(subset)
        # print(self.data_list)

        # 初始化 ind 到 文件名 的映射
        self._init_dataset()

        #
        print(f'{self.__class__.__name__} dataset length = {self.__len__()}')


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

        if DEBUGGING:
            for cl in range(self.nclasses):
                if cl == 1:
                    self.data_list[subset][cl] = glob.glob(folder + self.search_line % (cl))
                else:
                    self.data_list[subset][cl] = []
            return

        for cl in range(self.nclasses):
            # print(f'folder = {folder}')
            self.data_list[subset][cl] = glob.glob(folder + self.search_line % (cl))

    def _init_dataset(self):
        # 遍历所有文件名
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
                    # print(f'file_name = {file_name}')
                    if_legal = self._check_if_can_form_stblock(file_name, start_frame)
                    if not if_legal:
                        break
                    # 能到这里，说明当前的 stblock 取法是合法的，则记到 self.dataset 中
                    self.dataset[self.subset].append((file_name, start_frame, class_number))
                    # print(f'len(self.dataset[self.subset]) = {len(self.dataset[self.subset])}')

                    # 往后滑，继续尝试
                    start_frame += STEP_TO_NEXT_STBLOCK  # 滑动几帧，继续取下一个 stblock

    def _check_if_can_form_stblock(self, file_name, start_frame):
        for mdlt in self.modality_list:
            data_sample = self.datasetTypes[mdlt]._load_file(file_name)

            if data_sample is None:
                # print('FILE NOT EXIST.')
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

    def __getitem__(self, ind):
        sample = {}

        file_name, start_frame, label = self.dataset[self.subset][ind]
        for mdlt in self.modality_list:
            data = self.datasetTypes[mdlt]._get_data_by_filename_and_startframe(file_name=file_name, start_frame=start_frame)
            if mdlt == 'video':
                # v3.0 将视频模态拆分为四各模态： r_color, r_depth, l_color, l_depth
                i = 0
                for video_mdlt in ['r_color', 'r_depth', 'l_color', 'l_depth']:
                    sample[video_mdlt] = data[i]
                    i += 1
            else:
                sample[mdlt] = data


        return sample, int(label)

    def __len__(self):
        return len(self.dataset[self.subset])