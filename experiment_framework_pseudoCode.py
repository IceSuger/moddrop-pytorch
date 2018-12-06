"""
"""
from script_generateDataset4M1 import generateLQDataset, generateDeltaStar
from testing_DataSelection_or_not import testWithoutDataSelection, testWithDataSelection
from training_Phi_s import readFilesAndFormTheDataframeAndWriteToDisk, trainAndTest_Phi_s

"""
1. 输入为高质量数据集，构造低质量数据集
包括完整的 train/valid/test。
"""
generateLQDataset(r=2, subset='train')
generateLQDataset(r=2, subset='valid')

print("=================================")
print("== 1. LQ dataset generated. =====")
print("=================================")
"""
2. 对得到的低质量数据集 D_R, 划分 train/valid/test
事实上，如果做了第一步，这一步是已经完成了的。
"""
# 跳过。

"""
3. 用 D_R_train 来训练 phi_r
即，训练 phi_r 时的数据来源，不再是原来的 DatasetMultimodal 了，而是新的 DatasetLowQualityMultimodal ！
读数据的部分，即 __getitem__ 需要针对新的数据组织结构来改动！
"""
# 从 training_script 里粘，改

"""
4. 数据评价，得到相应的 QoU
这个也在第1步顺便做了。
"""
# 跳过。

"""
5. 生成 D_Q 的 label，即 delta_star

"""
generateDeltaStar(r=1, train_valid_test='train', path_D_R_root='D_R')
generateDeltaStar(r=1, train_valid_test='valid', path_D_R_root='D_R')

print("=================================")
print("== 5. DeltaStar generated. ======")
print("=================================")
"""
6. 训练 phi_s
"""
# 调用脚本 training_Phi_s.py 中的函数
df = readFilesAndFormTheDataframeAndWriteToDisk(path_D_R_root='D_R', result_file_name = 'QoU_2_deltaStar.csv')
clf = trainAndTest_Phi_s(df)

print("=================================")
print("== 6. Phi_s trained. ============")
print("=================================")

"""
7. 跑两趟：
    7.1 直接将 D_R_valid 送入 phi_r
    7.2 用 DatasetSelectedMultimodal 读取 D_R_valid 送入 phi_r
        记得在 DatasetSelectedMultimodal 的 __init__ 中，加载训练好的 phi_s
"""
result_no_data_selection = testWithoutDataSelection()
result_with_data_selection = testWithDataSelection(clf)
print('不经过数据选择模块，测试准确率：', result_no_data_selection)
print('经过数据选择模块，测试准确率：', result_with_data_selection)

print("=================================")
print("== 7. Experiment finished. ======")
print("=================================")