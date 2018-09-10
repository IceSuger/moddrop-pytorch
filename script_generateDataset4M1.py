"""
1. generate damaged multimodal dataset and QoUs over this dataset
2. generate labels
"""

# 1.

"""
for s in D0:
    for u in range(r):  # randomly get r damaged data sample
        _s = {}
        QoU = []
        for mdlt in s:
            for dmg_dimension in dmg_functions:
                _s[mdlt] = dmg_func(s[mdlt])
            
            for score_func in score_funcs:
                QoU.append(score_func(_s[mdlt]))
        
        # save to disk
        cPickle(_s, QoU, label_of_s, u, if_s_name_available_then_set_it_here)
"""

# 2.

"""
for _s, QoUs in DataLoader(dataset_generated_above):
    s_probs = set()
    Set_modality = getSubsets([modalities])
    for subset in Set_modality:
        probs, result = M0(mask(_s, subset))
        Set_probs.append((probs, result, label, subset))
        
        #if result correct:  # 只有分类正确的前提下，熵低才有意义
               # 那么存在一个情况，可能分类就都不正确啊...
    
    sort Set_probs by (-correct, H(probs)) # ascend
    # 取第一个
    
    # if Set_probs is empty:
    #     subset_best = 
    # else:
    #     subset_best = argminH(Set_probs)
    
    # save to disk
    cPickle(QoUs, subset_best)
    
    # or save in memory
    df(dtypes = {'subset_best': str, others: float})
"""

# 3. Train M1
#
# M1 实现为 nn.Module 子类，两层神经网络
"""

"""

# 4. Use M1
#
# 输入
#   s: dict, 多模态数据
# 输出
#   result: 手势识别结果（类别）
"""
QoU = []
for mdlt in s:
    for score_func in score_funcs:
        QoU.append(score_func(_s[mdlt]))
subset_best = M1(QoU)
masked_s = mask(s, subset_best)    # subset: list, s: dict(key: mdlt)

result = M0(masked_s)
"""