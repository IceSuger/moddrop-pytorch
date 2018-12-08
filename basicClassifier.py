import torch
import collections
import numpy
import time

from datasetsOfLowQualityData.datasetLQAudio import DatasetLQAudio
from datasetsOfLowQualityData.datasetLQSkeleton import DatasetLQSkeleton
from datasetsOfLowQualityData.datasetLQVideoClassifier import DatasetLQVideoClassifier
from datasetsOfLowQualityData.datasetLQVideoFeatureExtractor import DatasetLQVideoFeatureExtractor
from datasetsOfLowQualityData.datasetOfDamagedMultimodal import DatasetOfDamagedMultimodal
from datasetsOfLowQualityData.datasetSelectedMultimodal import DatasetSelectedMultimodal
from torch import nn
from torch.utils.data import DataLoader
# from torch.autograd import Variable as V
from utils.visualize import Visualizer
import pandas as pd
from entropyEvaluating import softmax

import os
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.


current_time=time.strftime("%Y%m%d_%H%M%S")+"_"+str(numpy.random.randint(100000))

class basicClassifier(object):
    def __init__(self, input_folder, filter_folder, number_of_classes=21,
				 step=4, nframes=5, batch_size=42, modality='mocap', pretrained=False
                 ):
        self.signature = current_time

        # Input parameters
        self.nclasses = number_of_classes
        self.step = step
        self.nframes = nframes
        self.seq_per_class = 700 # 200

        self.modality = modality
        self.hand_list = collections.OrderedDict()
        self.input_size = {}
        self.params = []

        self.dataset = {}
        self.dataset['train'] = {}
        self.dataset['valid'] = {}
        self.dataset['test'] = {}
        self.data_list = {}

        # # Theano variables
        # tensor4 = T.TensorType(theano.config.floatX, (False,) * 4)
        # tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)

        # Network parameters
        self.conv_layers = []
        self.pooling = []
        self.fc_layers = []
        self.dropout_rates = []
        self.activation = 'relu'
        self.use_bias = True
        self.mask_weights = None
        self.mask_biases = None

        # Training parameters
        # lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results
        # self.batch_size = 42
        self.pretrained = pretrained
        # self.learning_rate_value = 0.05
        # self.learning_rate_decay = 0.999
        # self.epoch_counter = 1

        # Paths
        self.search_line = "*_g%02d*.pickle"
        self.input_folder = input_folder
        self.train_folder = self.input_folder + modality + '/train/'
        self.valid_folder = self.input_folder + modality + '/valid/'
        self.test_folder = self.input_folder + modality + '/test/'
        self.filter_folder = filter_folder
        self.filters_file = filter_folder + modality + 'Classifier_step' + str(step) + '.npz'

        self.modality = modality

        # 决定训练/验证/测试时是否可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def train_torch(self, datasetTypeCls, learning_rate_value=None, learning_rate_decay=None, num_epochs=5000, early_stop_epochs=20):
        """

        :param datasetTypeCls:
        :param learning_rate_value:
        :param learning_rate_decay:
        :param num_epochs:
        :param early_stop_epochs: 连续这么多个epoch上，val_loss都没有降低，则提前终止训练
        :return:
        """
        # Load dataset

        self.saved_params = []



        if self.pretrained:
            print('Saved model found. Loading...')
            self.load_model()

        if learning_rate_value is None:
            learning_rate_value = self.learning_rate_value
        if learning_rate_decay is None:
            learning_rate_decay = self.learning_rate_decay

        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")

        # print(self.sinputs) # ??????? 弄啥嘞？


        # [Xiao]
        vis = Visualizer('xiao-moddrop')

        # step 1: setup model
        model = self.model
        # model = model.cuda()

        # step 2: data\
        # train_data = DatasetVideoFeatureExtractor(self.input_folder, self.modality, 'train', self.hand_list, self.seq_per_class,
        #                                           self.nclasses, self.input_size, self.step, self.nframes)
        # val_data = DatasetVideoFeatureExtractor(self.input_folder, self.modality, 'valid', self.hand_list, 200,
        #                                         self.nclasses, self.input_size, self.step, self.nframes)
        if datasetTypeCls in [DatasetOfDamagedMultimodal, DatasetLQAudio,  DatasetLQVideoClassifier, DatasetLQVideoFeatureExtractor, DatasetLQSkeleton]:
            # print(f'In basicClassifier.py, df = {df}')
            train_data = datasetTypeCls(self.input_folder, train_valid_test='train')
            val_data = datasetTypeCls(self.input_folder, train_valid_test='valid')
        else:
            train_data = datasetTypeCls(self.input_folder, self.modality, 'train', self.hand_list,
                                                      self.seq_per_class,
                                                      self.nclasses, self.input_size, self.step, self.nframes)
            val_data = datasetTypeCls(self.input_folder, self.modality, 'valid', self.hand_list, 200,
                                                    self.nclasses, self.input_size, self.step, self.nframes)

        print('Dataset prepared.')

        # self._load_dataset('train')  # ？？
        train_loader = DataLoader(train_data, batch_size=42, shuffle=True, num_workers=56)  # num_workers 按 CPU 逻辑核数目来。查看命令是： cat /proc/cpuinfo| grep "processor"| wc -l
        val_loader = DataLoader(val_data, batch_size=42, shuffle=False, num_workers=56)

        print('DataLoader prepared.')
        # val_loader = DataLoader(self.val_data, 32)

        # step 3: criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 0.02 # 0.001
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1-0.9998, nesterov=True, momentum=0.8)

        # visdom show line of loss
        win = vis.line(
            X=numpy.array([0, 1]),
            Y=numpy.array([0, 1]),
            name="loss"
        )
        win1 = vis.line(
            X=numpy.array([0, 1]),
            Y=numpy.array([0, 1]),
            name="loss_epoch"
        )

        # step 4: go to GPU
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 这一句放到init方法中了
        model.to(self.device)


        print('Training begin...')
        # for data in train_loader:
        #     print(data[1])
        #     # break
        # print('HH=======25-20947489THRLGIHRSGHRNKGNREOSG========')
        best_val_loss = self.nclasses
        epochs_no_better_val_loss = 0

        for epoch in range(num_epochs):


            # In each epoch, we do a full pass over the training data:
            losses = []

            for ii, (data, label) in enumerate(train_loader):
                input = data
                target = label.to(torch.int64)

                # [Xiao] 如果是多模态的输入，需要区别对待
                if not isinstance(input, dict):
                    # 若当前的输入是 torch.tensor ，说明不是最终的多模态输入，可以直接上GPU
                    input, target = input.to(self.device), target.to(self.device)
                else:
                    # 若是字典，说明是多模态输入，这里先将label上GPU，其他的部分在输入model.forward()中分别取出来作为tensor再上GPU
                    target = target.to(self.device)

                # print(f'input.shape is : {input.shape}')
                # print(f'target.shape is : {target.shape}')

                # Create a loss expression for training, i.e., a scalar objective we want
                # to minimize (for our multi-class problem, it is the cross-entropy loss):
                self.optimizer.zero_grad()
                score = model(input)
                # print(f'score shape is: {score.shape}')
                loss = self.criterion(score, target)    # score 即为长度等于 nclasses 的“概率”向量， target 即为单一值（类别的编号）

                loss.backward()
                self.optimizer.step()

                losses.append(loss.data)



                # print(f'score is : {score}')
                # print(f'target is : {target}')

                # print(f'loss.data is: {loss.data}')
                # print(f'numpy.array(loss.data) is : {numpy.array(loss.data)}')

                # if ii % 10 == 0:
                #     vis.plot('loss', loss)

                # vis.line(X=torch.Tensor([ii + epoch*len(train_loader)]), Y=torch.Tensor([loss]), win=win, update='append', name='train_loss')


            # 计算验证集上的指标及可视化
            val_loss = self.val(model, val_loader)
            # 若验证误差降低，更新最好模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                epochs_no_better_val_loss = 0
            else:
                # 若验证loss没有降低，则累计到 early_stop_epochs 个epoch之后，就提前停止
                epochs_no_better_val_loss += 1
                if epochs_no_better_val_loss >= early_stop_epochs:
                    break
            # vis.plot('val_loss', val_loss)

            vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([sum(losses) / len(losses)]), win=win1, update='append',
                     name='mean_train_loss_per_epoch')
            vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([val_loss]), win=win1, update='append',
                     name='val_loss')

            vis.log("[Train Loss] epoch:{epoch},lr:{lr},loss:{loss}".format(
                epoch=epoch, loss=loss.data,
                lr=self.lr))
            vis.log("[Valid Loss] epoch:{epoch},lr:{lr},loss:{loss}".format(
                epoch=epoch, loss=val_loss,
                lr=self.lr))


    def _prepare_inputs(self, subset, ind):
        """
        Function to sample and concatenate inputs for each minibatches,
        used with pickle files.

        :type subset: str
        :param subset: data subset ("train", "valid" or "test")

        :type ind: int
        :param ind: minibatch index
        """

        inputs = []

        # Append data from all channels

        for hnd in self.hand_list:
            for mdlt in self.hand_list[hnd]:
                inputs.append(self.dataset[subset][hnd][mdlt][ind * self.batch_size:
                                                              (ind + 1) * self.batch_size])
        if subset in ['train', 'valid']:
            inputs.append(self.dataset[subset]['labels'][ind * self.batch_size:
                                                         (ind + 1) * self.batch_size])
        return inputs
        # this is a list of tuples of size (batch_size, channel=1, input_size)


    def val(self, model, dataloader):
        """
        计算模型在验证集上的准确率等信息
        """

        # 把模型设为验证模式
        model.eval()

        losses = []

        for ii, data in enumerate(dataloader):
            input, label = data
            if not isinstance(input, dict):
                val_input = input.to(self.device)
            else:
                val_input = input
            val_label = label.to(self.device)
            score = model(val_input)
            losses.append(self.criterion(score, val_label).data)

        # 把模型恢复为训练模式
        model.train()

        loss = sum(losses) / len(losses)
        return loss

    def test_torch(self, datasetTypeCls, phi_s=None, df=None):
        if datasetTypeCls == DatasetOfDamagedMultimodal or datasetTypeCls == DatasetSelectedMultimodal:
            # print(f'In basicClassifier.py, df = {df}')
            test_data = datasetTypeCls(self.input_folder, train_valid_test='valid', phi_s=phi_s, QoU2delta_df=df)
        else:
            test_data = datasetTypeCls(self.input_folder, self.modality, 'valid', self.hand_list, 589,
                                  self.nclasses, self.input_size, self.step, self.nframes, phi_s)
        # test_data = datasetTypeCls(self.input_folder, self.modality, 'train', self.hand_list, 700,
        #                            self.nclasses, self.input_size, self.step, self.nframes)
        # test_loader = DataLoader(test_data, batch_size=42, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        self.model.to(self.device)
        # 把模型设为验证模式
        self.model.eval()

        correct = 0
        total = 0
        predicted_to_save = numpy.zeros((1,21))
        labels_to_save = []
        with torch.no_grad():
            # for data in test_loader:
            #     images, labels = data
            #     outputs = net(images)
            #     _, predicted = torch.max(outputs.data, 1)
            #     total += labels.size(0)
            #     correct += (predicted == labels).sum().item()

            for ii, data in enumerate(test_loader):
                input, label = data
                if not isinstance(input, dict):
                    val_input = input.to(self.device)
                else:
                    val_input = input
                val_label = label.to(self.device)
                score = self.model(val_input)
                # print(f'score is: {score.data}')
                # print(f'score.data.shape is: {score.data.shape}')
                predicted = torch.argmax(score.data, dim=1)
                # predicted_to_save.extend(score.data.cpu().numpy())  # For saving as csv, as the features into RF.
                predicted_to_save = numpy.concatenate((predicted_to_save, softmax(score.data.cpu().numpy())), axis=0)
                # print(f'score.data is: \n{score.data}')
                # print(f'score.data is: \n{score.data.cpu().numpy()}')
                # print(f'score.data shape is: \n{score.data.cpu().numpy().shape}')
                print(f'predicted_to_save shape is : {predicted_to_save.shape}')
                total += label.size(0)
                labels_to_save.extend(label.data.numpy())
                correct += sum(predicted == val_label).item()
                # losses.append(self.criterion(score, val_label).data)

        # print('Accuracy of the network on valid set as the test set: %d %%' % (
        #         100 * correct / total))
        print(f'Accuracy over the {total} samples in validset(as testset) is {(100 * correct / total)}')

        # Save as csv
        # pd.Series(predicted_to_save).to_csv('results/'+ 'valid_prob/' + 'predicted_'+self.model_name+'.csv')
        # pd.Series(labels_to_save).to_csv('results/'+ 'valid_prob/' + 'labels_' + self.model_name + '.csv')
        # df_to_save = pd.DataFrame(pd.concat([pd.DataFrame(predicted_to_save), pd.DataFrame(labels_to_save)], axis=1))
        # df_to_save.to_csv('results/'+self.model_name)
        pd.DataFrame(predicted_to_save[1:], columns=range(21)).to_csv('results/' + 'valid_prob/' + 'predicted_' + self.model_name + '.csv')
        pd.Series(labels_to_save).to_csv('results/' + 'valid_prob/' + 'labels_' + self.model_name + '.csv')

        # 返回准确率
        return (100 * correct / total)


    def test_torch_modality_selection(self, datasetTypeCls):
        test_data = datasetTypeCls(self.input_folder, self.modality, 'valid', self.hand_list, 589,
                                  self.nclasses, self.input_size, self.step, self.nframes)
        # test_data = datasetTypeCls(self.input_folder, self.modality, 'train', self.hand_list, 700,
        #                            self.nclasses, self.input_size, self.step, self.nframes)
        test_loader = DataLoader(test_data, batch_size=42, shuffle=False, num_workers=56)
        self.model.to(self.device)

        # 把模型设为验证模式
        self.model.eval()

        correct = 0
        total = 0
        predicted_to_save = numpy.zeros((1,21))
        labels_to_save = []
        with torch.no_grad():
            # for data in test_loader:
            #     images, labels = data
            #     outputs = net(images)
            #     _, predicted = torch.max(outputs.data, 1)
            #     total += labels.size(0)
            #     correct += (predicted == labels).sum().item()

            for ii, data in enumerate(test_loader):
                input, label = data
                if not isinstance(input, dict):
                    val_input = input.to(self.device)
                else:
                    val_input = input
                val_label = label.to(self.device)
                score = self.model(val_input)
                # print(f'score is: {score.data}')
                # print(f'score.data.shape is: {score.data.shape}')
                predicted = torch.argmax(score.data, dim=1)
                # predicted_to_save.extend(score.data.cpu().numpy())  # For saving as csv, as the features into RF.
                predicted_to_save = numpy.concatenate((predicted_to_save, softmax(score.data.cpu().numpy())), axis=0)
                # print(f'score.data is: \n{score.data}')
                # print(f'score.data is: \n{score.data.cpu().numpy()}')
                # print(f'score.data shape is: \n{score.data.cpu().numpy().shape}')
                print(f'predicted_to_save shape is : {predicted_to_save.shape}')
                total += label.size(0)
                labels_to_save.extend(label.data.numpy())
                correct += sum(predicted == val_label).item()
                # losses.append(self.criterion(score, val_label).data)

        # print('Accuracy of the network on valid set as the test set: %d %%' % (
        #         100 * correct / total))
        print(f'Accuracy over the {total} samples in validset(as testset) is {(100 * correct / total)}')

        # Save as csv
        # pd.Series(predicted_to_save).to_csv('results/'+ 'valid_prob/' + 'predicted_'+self.model_name+'.csv')
        # pd.Series(labels_to_save).to_csv('results/'+ 'valid_prob/' + 'labels_' + self.model_name + '.csv')
        # df_to_save = pd.DataFrame(pd.concat([pd.DataFrame(predicted_to_save), pd.DataFrame(labels_to_save)], axis=1))
        # df_to_save.to_csv('results/'+self.model_name)
        pd.DataFrame(predicted_to_save[1:], columns=range(21)).to_csv('results/' + 'valid_prob/' + 'predicted_' + self.model_name + '.csv')
        pd.Series(labels_to_save).to_csv('results/' + 'valid_prob/' + 'labels_' + self.model_name + '.csv')



    def save_model(self, name = None):
        if name is None:
            name = 'checkpoints/' + self.model_name + '.pth'
        torch.save(self.network.state_dict(), name)
        return name

    def load_weights(self, name = None):
        if name is None:
            name = 'checkpoints/' + self.model_name + '.pth'

        extractor_dict = torch.load(name)
        self.network.load_state_dict(extractor_dict)
        # self.video_network = nn.Sequential(OrderedDict(list(self.video_network.named_children())))