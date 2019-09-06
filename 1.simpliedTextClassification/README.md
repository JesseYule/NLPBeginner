### 任务一：基于机器学习的文本分类

实现基于logistic的文本分类

1. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

2. 知识点：
   1. 文本特征表示：[词袋模型](https://jesseyule.github.io/naturallanguage/bow/content.html)
   2. 分类器：[logistic回归](https://jesseyule.github.io/machinelearning/logisticRegression/content.html)、损失函数、[梯度下降](https://jesseyule.github.io/machinelearning/gradientDescent/content.html)、[特征选择](https://jesseyule.github.io/machinelearning/featureEngineering/content.html)
   3. [交叉检验](https://jesseyule.github.io/machinelearning/crossValidation/content.html)

3. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 

4. [问题简单分析](https://jesseyule.github.io/naturallanguage/simplifiedTextClassification/content.html)

### 代码说明

​	classify.py只针对部分数据进行分析，但是模型是完整应用了词袋模型以及logistic模型，在此代码的基础上，可以改进文本特征的表示方法，比如采用二元特征表示等等，另一方面logistic模型等等也可以进行相应改进。

