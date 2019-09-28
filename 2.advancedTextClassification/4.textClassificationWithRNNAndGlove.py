from __future__ import unicode_literals, print_function, division
from io import open
import glob
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np
import random


# 第一步，加载GloVe

def loadGlove(word2vec_file):
    glove_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    return glove_model


# 第二步，加载数据

def findFiles(path):
    return glob.glob(path)


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]


# 第三步，正式构建循环神经网络
# 主要构建了一个线性隐层和一个线性输出层

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # 隐层包含正常神经元（i2o）和虚神经元（i2h）
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        combined = torch.cat((input, hidden), 1)  # 输入包括当前输入和以前的隐藏状态
        hidden = self.i2h(combined)  # 更新hidden层，留给下一次训练当作输入
        output = self.i2o(combined)  # 隐层输出
        output = self.softmax(output)  # 对隐层输出作softmax（即输出层激活函数）
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)  # 初始化虚神经元


# 第四步，构建一些辅助训练的函数
# 以下函数主要分析输出结果对应的sentiment得分

def resultFromOutput(output):
    top_n, top_i = output.topk(1)  # topk函数可得到最大值在结果中的位置索引
    return top_n, top_i


# 第四步，正式训练神经网络

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel(result, sentence):

    hidden = rnn.initHidden()

    model_output = torch.ones(1, 5)

    rnn.zero_grad()

    # 重新处理输入的分类结果，把其变为pytorch可处理的类型
    result = int(result)
    result = torch.tensor([result]).long()  # 为什么这里要Long类型，明明其他都是int64

    # 重新处理输入的句子，主要把句子变为单词的列表，并且去掉单词之间的空格
    input_sentence = []
    for char in sentence.split(' '):
        input_sentence.append(char.strip())

    # 将句子分为一个个单词进行输入
    for k in range(len(input_sentence)):
        test = glove_model['test']
        try:
            input_char = glove_model[input_sentence[k]]  # 将单词转化为词向量
            input_char = np.mat(input_char)  # 改变向量格式，把一维数组改为1*len(input)的二维矩阵，这是pytorch要求的输入格式
            input_char = torch.from_numpy(input_char).float()  # 注意数据类型

            model_output, hidden = rnn(input_char, hidden)
            # print(model_output)
            # print(result)
        except Exception as e:
            input_char = torch.zeros(1, len(test)).float()  # 假如GloVe中没有对应的单词，直接用全0向量代替
            model_output, hidden = rnn(input_char, hidden)
            # print(model_output)
            # print(result)
            # print(e)

    loss = criterion(model_output, result)
    loss.backward()

    # 将参数的梯度添加到其值中，乘以学习速率
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return model_output, loss.item()


if __name__ == '__main__':

    # 加载glove预训练词向量
    word2vec_file = r'../../../glove/glove.42B.300d.word2vec.txt'
    glove_model = loadGlove(word2vec_file)

    # 句子情感的可能输出数值
    result = [0, 1, 2, 3, 4]

    # 定义神经网络

    current_loss = 0
    all_losses = []
    n_hidden = 128
    input_size = 300
    output_size = 5
    rnn = RNN(input_size, n_hidden, output_size)

    # 读取数据以及进行预处理

    lines = readLines('data/train.tsv')
    sentence = []
    result = []
    for i in range(len(lines)-1):
        sentence.append(lines[i+1][4:-1])  # 读取每一行数据中对应的文本的列
        result.append(lines[i+1][-1])  # 读取每一行数据中对应的分类结果的列

    # 正式训练，注意，这里只是按顺序抽取数据进行训练

    criterion = nn.NLLLoss()
    learning_rate = 0.001

    # 跟踪绘图的损失
    print_every = 5000
    plot_every = 1000

    start = time.time()

    iter_max = 100000

    for j in range(iter_max):

        rand_j = random.randint(1, len(lines)-2)  # 随机训练

        output, loss = trainModel(result[rand_j], sentence[rand_j])
        current_loss += loss
        try:
            # 打印迭代的编号，损失，名字和猜测
            if j % print_every == 0:
                guess, guess_i = resultFromOutput(output)
                check_guess = int(guess)
                check_guess = check_guess * (-1)
                check_result = int(result[rand_j])
                print('guess: ', check_guess)
                print('result: ', check_result)
                correct = '✓' if check_guess == check_result else '✗ (%s)' % check_result
                print('%d   (%s) %.4f  %s / %s  %s' % (
                j, timeSince(start), loss, sentence[rand_j], check_guess, correct))
        except Exception as e:
            print(e)
            print('rand_j: ', rand_j)
            print('output: ', output)
            continue

        # 将当前损失平均值添加到损失列表中
        if j % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()

