### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系

1. 数据集：

   * https://nlp.stanford.edu/projects/snli/

   * https://www.nyu.edu/projects/bowman/multinli/

2. 知识点：

   1. [LSTM](<https://jesseyule.github.io/machinelearning/lstm/content.html>)
   2. [seq2seq](<https://jesseyule.github.io/naturallanguage/seq2seq/content.html>)
   3. [注意力机制](<https://jesseyule.github.io/naturallanguage/attentionMechanism/content.html>)
   4. [ESIM](<https://jesseyule.github.io/naturallanguage/ESIM/content.html>)

### 代码说明

​	本次实验主要是基于SNLI和MultiNLI这两个语料库进行的文本匹配，关于这两个语料库的说明和实验目的在网址中有详细介绍。

​	实验主要是利用ESIM模型进行的文本匹配，关于ESIM模型的详细介绍分析在知识点中我也总结了，但是我仍然强烈建议阅读papers中的几篇论文，它们都介绍了如何在文本匹配问题中应用ESIM。

​	另外，代码被我大幅度简化，主要是为了可以清晰地研究整个模型的思路，完整的代码请看：<https://github.com/nyu-mll/multiNLI>

​	





