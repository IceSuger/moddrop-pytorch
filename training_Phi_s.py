import pandas as pd
import os
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing


def readFilesAndFormTheDataframeAndWriteToDisk(path_D_Q_root = 'D_Q', train_valid_test='train', result_file_name = 'QoU_to_deltaStar.csv'):
    root = '/home/xiaoyunlong/code/moddrop-pytorch/' + path_D_Q_root + '/' + train_valid_test + '/'
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
    print(df.head())
    print(df.describe())

    # 将 label 中的空列表（即对应于生成delta_star的过程中，phi_r没能正确分类的那些样本），都替换为模态全集
    origin_label_column_number = len(df.columns) - 1
    df[origin_label_column_number].fillna('ALL_MODAL')
    print(df.head())
    print(df.describe())

    # 将 label 转为数值，加在最后一列
    df['cc'] = pd.Categorical(df[origin_label_column_number])
    df['code'] = df.cc.cat.codes
    print(df.head())
    print(df.describe())

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

    label_column_number = len(df.columns) - 1
    last_feature_column_number = label_column_number - 2
    print(f'label_column_number = {label_column_number}, last_feature_column_number = {last_feature_column_number}')

    # X = df.iloc[:train_len,:9]
    # y = df.iloc[:train_len,11]
    # test_X = df.iloc[train_len:,:9]
    # test_y = df.iloc[train_len:,11]

    X = df.iloc[:train_len, :last_feature_column_number]
    y = df.iloc[:train_len, label_column_number]
    test_X = df.iloc[train_len:, :last_feature_column_number]
    test_y = df.iloc[train_len:, label_column_number]

    # 做标准化
    scaler = preprocessing.Normalizer()
    X = scaler.fit_transform(X)
    test_X = scaler.transform(test_X)

    clf = GradientBoostingClassifier()
    # clf = RandomForestClassifier()
    clf.fit(X, y)

    print(clf.feature_importances_)

    pred_y = clf.predict(test_X)

    rightCnt = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y.iloc[i]:
            rightCnt += 1
    print(f"测试集上准确率为 {rightCnt / len(pred_y)}")   # 输出测试集上 phi_s 的准确率

    # v3.0.1 看看训练集上的准确率，虽然意义不大，但打出来看看。。
    pred_y_train = clf.predict(X)

    rightCnt = 0
    for i in range(len(pred_y_train)):
        if pred_y_train[i] == y.iloc[i]:
            rightCnt += 1
    print(f"训练集上准确率为 {rightCnt / len(pred_y_train)}")  # 输出训练集上 phi_s 的准确率

    return clf