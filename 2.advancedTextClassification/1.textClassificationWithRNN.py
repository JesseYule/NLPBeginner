from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 第一步，数据预处理
# 因为数据几乎都是罗马化的文本，所以要将其从unicode转化为ASCII


def findFiles(path):
    return glob.glob(path)


all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)


# 将unicode转成ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


category_lines = {}
all_categories = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines  # 一个字典变量储存每一种语言及其对应的每一行文本（名字）列表的映射关系

n_categories = len(all_categories)

# 以上处理可得到category_line，它保存了语种-姓名列表，也有all_categories保存语种列表，以及n_categories表示语种数量

#  下面的步骤就是把word embedding，因为是分析单词，所以主要通过独热编码表示字母，再根据单词长度构建相应维度的tensor
#  注意，这里不同长度单词的矩阵维度也不同，额外的一维是batch的维度


#  找到字母在字母表中的位置
def letterToIndex(letter):
    return all_letters.find(letter)


# 独热编码，调用letterToIndex，将字母转化为一个tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# 根据单词长度将一个个字母的tensor构建成表示一个单词的<line_length*1*n_letters>的tensor
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# 第二步，正式构建循环神经网络
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


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = lineToTensor('Hofler')

hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)


# 第三步，构建一些辅助函数辅助训练
# 以下函数主要分析输出结果对应哪种语言（只输出可能性最大的结果）


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)  # topk函数可得到最大值在结果中的位置索引
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


# 以下函数是关于随机选择训练样本
def randomChoice(l):
    return l[random.randint(0, len(l)-1)]


def randomTrainingExample():
    category = randomChoice(all_categories)  # 在所有种类中随机选择一种
    line = randomChoice(category_lines[category])  # 在选中的语言中再随机选一个名字

    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)  # 保存语种的index
    line_tensor = lineToTensor(line)  # 把名字转化为tensor
    return category, line, category_tensor, line_tensor


# 第四步，正式训练神经网络

criterion = nn.NLLLoss()  # 定义损失函数
learning_rate = 0.005


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    # 下面是训练一个单词的过程，注意这里是针对一个单词的一个个字符进行输入
    # 对RNN来说，完整的一次训练是完整输入一个单词的所有字符的过程
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)


    loss = criterion(output, category_tensor)
    loss.backward()

    # 将参数的梯度添加到其值中，乘以学习速率
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


n_iters = 100000
print_every = 5000
plot_every = 1000

# 跟踪绘图的损失
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()


#  正式进行批量训练，针对随机选择的大量单词训练RNN
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # 打印迭代的编号，损失，名字和猜测
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        print('guess: ', guess)
        print('category: ', category)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d  %d%% (%s) %.4f  %s / %s  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 将当前损失平均值添加到损失列表中
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()
