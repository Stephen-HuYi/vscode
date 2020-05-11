'''
@Description: 
@Author: HuYi
@Date: 2020-04-29 16:44:47
@LastEditors: HuYi
@LastEditTime: 2020-04-29 16:47:41
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
seqs = [np.array([[0], [0], [1], [1], [2], [2], [3], [3]]),  # AABBCCDD
        np.array([[0], [1], [1], [2], [1], [1], [3], [3]]),  # ABBCBBDD
        np.array([[0], [2], [1], [2], [1], [2], [3]])]      # ACBCBCD

# 学习HMM 模型参数
exp_hmm.train_batch(seqs)
# 输出模型参数
print("start prob:", exp_hmm.start_prob)
print("transition matrix", exp_hmm.transmat_prob)
print("observation probability matrix", exp_hmm.emission_prob)

# 输入测试序列
seq_test = np.array([[0], [1], [1], [1], [2], [3], [3], [3]])  # ABBBCDDD
# 输出测试序列出现概率的对数
logprob = exp_hmm.X_prob(seq_test)  # 问题B
prob = np.exp(logprob)

print("logprob: ", logprob)
print("prob of seq_test: ", prob)
