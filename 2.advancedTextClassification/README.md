### 任务二：基于深度学习的文本分类

熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

1. 参考

   1. https://pytorch.org/
   2. Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>
   3. <https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>

2. word embedding 的方式初始化

3. 随机embedding的初始化方式

4. 用glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/

5. 知识点：

   1. [卷积神经网络](<https://jesseyule.github.io/machinelearning/cnn/content.html>)
   2. [循环神经网络](<https://jesseyule.github.io/machinelearning/rnn/content.html>)
   3. [word2vec](<https://jesseyule.github.io/naturallanguage/word2vec/content.html>)
   4. [GloVe](<https://jesseyule.github.io/naturallanguage/gloVe/content.html>)

   

### 代码说明

​	1.textClassificationWithRNN.py和2.GloVe.py是把RNN应用到文本分类以及GloVe的简单使用实例，基于这两个代码的基础上，把输入数据换成我们要分析的情感数据，就得到3.textClassificationWithRNNAndGlove.py。在改写的过程中，主要注意数据的格式，因为pytorch对输入数据的格式有很严格的要求（比如数据的维数），所以必须检查清楚避免出错，建议在理解第一第二个文件的代码的基础上，自行改写出第三个文件。



### 结果分析

![result](/Users/junjieyu/Documents/programming/github_projects/NLPBeginner/2.advancedTextClassification/result/result.png)

 ![plot](/Users/junjieyu/Documents/programming/github_projects/NLPBeginner/2.advancedTextClassification/result/plot.png)

​	从结果可以看出，其实模型训练效果并不算好，主要原因有以下几点：

1. 模型是按顺序训练数据，实际上应该进行随机抽取数据进行训练
2. 模型只有一层隐层，这也可能导致模型训练效果欠缺
3. GloVe缺失部分词向量，对这些词向量模型里都以全0向量代替，对模型的结果也可能造成影响

