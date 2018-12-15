"""

"""
import gc

from CONSTS import R, PATH_D_Q_ROOT
from script_generateDataset4M1 import generateLQDataset, generateDeltaStar, generateLQDataset_for_experiment1, \
    generateLQDataset_for_experiment2
from testing_DataSelection_or_not import testWithoutDataSelection, testHQWithoutDataSelection
from training_Phi_R import trainingLQClassifier
from training_Phi_s import readFilesAndFormTheDataframeAndWriteToDisk, trainAndTest_Phi_s

"""
1. 用高质量数据集训练 phi_r 
    1.1 用原数据集训练 phi_r 
    1.2 随机破坏，从而得到各种质量的数据
    1.3 用各种质量的数据，送进 phi_r 得到相应的 delta_star，即得到了 D_Q
    1.4 用 D_Q 训练 phi_s
"""
# print("=================================")
# print("== 1.1 Phi_r training. ===========")
# print("=================================")
# # 先挨个训练单模态的组件
# # 再训练多模态的 phi_r
# mdlts = [# 'skeleton',
#          # 'videoFeat',
#          # 'video',
#          # 'audio',
#          # 'multimodal',
#          'moddropMultimodal']
# for mdlt in mdlts:
#     trainingLQClassifier(cl_mode=mdlt)
#     print("======================================")
#     print(f"== 1.1 {mdlt} of Phi_r is trained. ==")
#     print("======================================")
#
# # 顺便测一下准确率
# result_no_data_selection = testHQWithoutDataSelection()
#
# print("=================================")
# print("== 1.1 Phi_r trained. ============")
# print("=================================")

########################################################################################

# print("=================================")
# print("== 1.2 LQ generating... =========")
# print("=================================")
#
# generateLQDataset(r=R, subset='train')
# generateLQDataset(r=R, subset='valid')
#
# print("=================================")
# print("== 1.2 LQ generated. =========")
# print("=================================")

########################################################################################

print("=================================")
print("== 1.3 DeltaStar generating..... =")
print("=================================")

generateDeltaStar(r=R, train_valid_test='train', path_D_Q_root=PATH_D_Q_ROOT)
generateDeltaStar(r=R, train_valid_test='valid', path_D_Q_root=PATH_D_Q_ROOT)

print("=================================")
print("== 1.3 DeltaStar generated. ======")

########################################################################################

print("=================================")
print("== 1.4 Phi_s training... =========")
print("=================================")

df = readFilesAndFormTheDataframeAndWriteToDisk(path_D_Q_root=PATH_D_Q_ROOT, train_valid_test='train', result_file_name = 'QoU_to_deltaStar.csv')
gc.collect()
clf = trainAndTest_Phi_s(df)

print("=================================")
print("== 1.4 Phi_s trained. ============")
print("=================================")


"""
【注意】phi_s 是用各种随机的破坏程度得到的各种各样的质量评分向量对应的 D_Q 训练出来的！且是固定的。

2. 遍历破坏方式 dmg_func in dmg_funcs：
        遍历破坏程度 degree = 0, 10, 20, ..., 100:
                遍历被污染模态子集 delta：
                        2.1 LQ_dataset = dmg_func(delta, degree, HQ_dataset)
                        2.2 不经过数据选择模块，跑 test
                        2.3 经过数据选择模块，跑 test
"""
generateLQDataset_for_experiment1(clf=clf, df=df)

"""
EXPERIMENT 2 
"""
# generateLQDataset_for_experiment2(clf=clf, df=df)