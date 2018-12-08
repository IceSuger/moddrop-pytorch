import pandas as pd
import os
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing


def readFilesAndFormTheDataframeAndWriteToDisk(path_D_R_root = 'D_R', train_valid_test='train', result_file_name = 'QoU_to_deltaStar.csv'):
    root = '/home/xiaoyunlong/code/moddrop-pytorch/' + path_D_R_root + '/' + train_valid_test + '/'
    search_line = '*'
    data = []

    # 获取文件列表
    file_list = glob.glob(root + search_line)

    # 遍历文件列表
    n = len(file_list)
    pct = n // 100
    i = 0

    for filename in file_list:
        if i % pct == 0:
            print(f'i: {i}, {i/n * 100}% files found.')
        i += 1

        file_path = os.path.join(root, filename)
        with open(file_path, 'rb') as f:
            s1 = pickle.load(f, encoding='iso-8859-1')

        # 写到 list 中，作为一行
        qou = list(map(lambda x: x.numpy()[0], s1['QoU']))
        qou.append(str(s1['subset_best'])[1:-1])
        data.append(qou)

    # 将 二维list 转为 df
    df = pd.DataFrame(data)
    # 将 label 转为数值，加在最后一列
    df['cc'] = pd.Categorical(df[9])
    df['code'] = df.cc.cat.codes

    # 写文件
    df.to_csv(result_file_name, index=False)

    return df


"""
用 sklearn 中的 RF 啥的跑跑看看
"""
def trainAndTest_Phi_s(df):
    """
    训练 phi_s ，测试并输出其准确率。
    :param df: 输入的大表，QoU -> deltaStar 的映射
    :return: 训练后的分类器
    """
    # 划分测试和验证集
    split_rate = 0.8
    n = len(df)
    train_len = int(n * split_rate)

    X = df.iloc[:train_len,:9]
    y = df.iloc[:train_len,11]
    test_X = df.iloc[train_len:,:9]
    test_y = df.iloc[train_len:,11]

    # 做标准化
    scaler = preprocessing.Normalizer()
    X = scaler.fit_transform(X)
    test_X = scaler.transform(test_X)

    clf = GradientBoostingClassifier()
    clf.fit(X, y)

    print(clf.feature_importances_)

    pred_y = clf.predict(test_X)

    rightCnt = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y.iloc[i]:
            rightCnt += 1
    print(rightCnt / len(pred_y))   # 输出测试集上 phi_s 的准确率

    return clf