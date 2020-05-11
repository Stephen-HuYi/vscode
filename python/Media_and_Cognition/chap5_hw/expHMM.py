'''
@Description:
@Author: HuYi
@Date: 2020-04-29 16:44:47
@LastEditors: HuYi
@LastEditTime: 2020-04-29 18:02:04
'''
# -*- coding:utf-8 -*-
# 博客: blog.csdn.net/tostq
import numpy as np
import hmm

state_num = 3
observation_num = 4
exp_hmm = hmm.DiscreteHMM(3, 4)
exp_hmm.start_prob = np.ones(3)/3.0
exp_hmm.transmat_prob = np.ones((3, 3))/3.0
exp_hmm.emission_prob = np.array([[.25, .25, .25, .25],
                                  [.3, .3, .3, .1],
                                  [.2, .2, .4, .4]])

exp_hmm.trained = True

# 输入训练序列
seqs1 = [np.array([[0], [0], [1], [1], [2], [2], [3], [3]]),  # AABBCCDD
         np.array([[0], [1], [1], [2], [1], [1], [3], [3]]),  # ABBCBBDD
         np.array([[0], [2], [1], [2], [1], [2], [3]]),  # ACBCBCD
         np.array([[0], [3]]),  # AD
         np.array([[0], [2], [1], [2], [1], [0], [
             1], [2], [3], [3]]),  # ACBCBABCDD
         np.array([[1], [0], [1], [0], [0], [3], [3], [3]]),  # BABAADDD
         np.array([[1], [0], [1], [2], [3], [2], [2]]),  # BABCDCC
         np.array([[0], [1], [3], [1], [1], [2], [2], [3], [3]]),  # ABDBBCCDD
         np.array([[0], [1], [0], [0], [0], [2], [
             3], [2], [2], [3]]),  # ABAAACDCCD
         np.array([[0], [1], [3]])]  # ABD
seqs2 = [np.array([[3], [3], [2], [2], [1], [1], [0], [0]]),  # DDCCBBAA
         np.array([[3], [3], [0], [1], [2], [1], [0]]),  # DDABCBA
         np.array([[2], [3], [2], [3], [2], [1], [0], [1], [0]]),  # CDCDCBABA
         np.array([[3], [3], [2], [2], [0]]),  # DDBBA
         np.array([[3], [0], [3], [0], [2], [1], [1], [0], [0]]),  # DADACBBAA
         np.array([[2], [3], [3], [2], [2], [1], [0]]),  # CDDCCBA
         np.array([[1], [3], [3], [1], [2], [0], [0], [0], [0]]),  # BDDBCAAAA
         np.array([[1], [1], [0], [1], [1], [3], [
                  3], [3], [2], [3]]),  # BBABBDDDCD
         np.array([[3], [3], [0], [3], [3], [1], [2], [0], [0]]),  # DDADDBCAA
         np.array([[3], [3], [2], [0], [0]])]  # DDCAA
# 学习HMM 模型参数
exp_hmm.train_batch(seqs1)
# 输出模型参数
print("start prob:", exp_hmm.start_prob)
print("transition matrix", exp_hmm.transmat_prob)
print("observation probability matrix", exp_hmm.emission_prob)

# 输入测试序列
seq_test = [np.array([[0], [1], [1], [1], [2], [3], [3], [3]]),  # ABBBCDDD
            np.array([[3], [0], [3], [1], [2], [1], [0], [0]]),  # DADBCBAA
            np.array([[2], [3], [2], [1], [0], [1], [0]]),  # CDCBABA
            np.array([[0], [3], [1], [1], [1], [2], [3]])]  # ADBBBCD
# 输出测试序列出现概率的对数
for i in range(4):
    logprob = exp_hmm.X_prob(seq_test[i])  # 问题B
    prob = np.exp(logprob)
    print("logprob", i, ": ",  logprob)
    print("prob of seq_test", i, ": ", prob)
