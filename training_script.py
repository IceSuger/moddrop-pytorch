#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#


__docformat__ = 'restructedtext en'

import os
import sys
os.environ['http_proxy'] = ''   # This line for preventing Visdom from not showing anything.

# from motionDetector import motionDetector
# from audioClassifier import audioClassifier
from skeletonClassifier import skeletonClassifier
from videoFeatureExtractor import videoFeatureExtractor
from videoClassifier import videoClassifier
# from multimodalClassifier import multimodalClassifier

from datasetVideoClassifier import DatasetVideoClassifier
from datasetVideoFeatureExtractor import DatasetVideoFeatureExtractor
from datasetSkeleton import DatasetSkeleton

''' Import the relevant classes from the util module.
	 - skeletonClassifier trains a 3D-ConvNet using mocap data
	 - videoClassifier trains a 3D convNet using video data of hands
	 - multimodalClassifier trains using both moCap and video data
	 - motionDetector trains a motion detector and used in post-training
	   to improve classification results.
	   This classifier implements late-fusion using a shared-hidden
	   layer
'''

# source_folder = '/mnt/data/dramacha/data_preprocessedv2/'
source_folder = '/home/xiaoyunlong/downloads/DeepGesture/Montalbano/'
'''
	Location of the dataset (Chalearn 2014) which has been pre-processed.
'''

filter_folder = 'filters/'

'''
	Filters refer to the saved-weights of the pre-trained
	convolutional neural network
'''

cl_methods = { 'skeleton' : skeletonClassifier,
              # 'motionDetector' : motionDetector,
              'video' : videoClassifier,
              'videoFeat': videoFeatureExtractor,
              # 'audio' : audioClassifier,
              # 'multimodal' : multimodalClassifier
              }

dataset_types = {
            'video' : DatasetVideoClassifier,
            'videoFeat': DatasetVideoFeatureExtractor,
            'skeleton': DatasetSkeleton,
            }


'''
	Select a classifier and temporal sampling 'step' size.
'''

if len(sys.argv) < 2:
    print("Usage: {0} classifier step".format(sys.argv[0]))
    exit(1)
else:
    cl_mode = sys.argv[1]
    if not cl_mode == 'motionDetector':
        step = int(sys.argv[2])
    else:
        step = 1
        # if motionDetector is selected, set step=1

if not cl_mode in cl_methods.keys():
    print('Unknown classifier. Options:', cl_methods.keys())
    exit(1)

do_lmdb = False
if len(sys.argv) > 3:
  do_lmdb = (sys.argv[3] == 'lmdb')

if not step in [1, 2, 3, 4]:
    print('"Step" can take the following values: [ 1 | 2 | 3 | 4 ]')
    exit(1)

''' Initialize a classifier
'''


#try:
classifier = cl_methods[cl_mode](step = step,
                                 input_folder = source_folder,
                                 filter_folder = filter_folder)
dataset_type_cls = dataset_types[cl_mode]
#except Exception as e:
#print 'Error: ',e

classifier.build_network() # build the model


#classifier.set_filters() # initialize with pretrained filters
#classifier.train_lmdb()#learning_rate_value=0.0)

classifier.train_torch(dataset_type_cls)

