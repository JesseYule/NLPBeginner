import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


def classify():
	df = pd.read_csv('../data/train.tsv', header=0, delimiter='\t')
	df = df[0:63]  # 为了简化只取第一句话作为例子
	x_train = df['Phrase']
	y_train = df['Sentiment']
	all = []

	#  构建词袋
	for i in range(len(x_train)):
		all.append(x_train[i])
	voc = set(all)  # 删除重复数据

	x_train_idx = []

#  将文本转化为向量形式
	for i in range(len(x_train)):
		tmp = np.zeros(len(voc))
		for j, word in enumerate(voc):  # 将voc转为索引序列，同时列出数据和下标
			tmp[j] = x_train[i].count(word)	  # 计算词袋中的每个词在句子中出现的次数，填入向量中
		x_train_idx.append(tmp)
	x_train_id = np.array(x_train_idx)

	logist = LogisticRegression()
	logist.fit(x_train_id, y_train)
	x_test = x_train_id  # 为了简化过程用回训练数据测试模型，实际上应该划分一个测试集
	predicted = logist.predict(x_test)
	print(np.mean(predicted == y_train))


classify()
