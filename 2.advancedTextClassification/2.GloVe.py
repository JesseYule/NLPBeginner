from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# 这里使用gensim工具包加载训练好的GloVe词向量，首先利用gensim把glove转换成方便gensim加载的word2vec格式
# 网上下载的训练好的glove词向量
glove_input_file = r'../../../glove/glove.42B.300d.txt'
# 指定转化为word2vec格式后文件的名称
word2vec_output_file = r'../../../glove/glove.42B.300d.word2vec.txt'
# 转换操作，注意，这个操作只需要进行一次
# glove2word2vec(glove_input_file, word2vec_output_file)

# 加载模型
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# 获得单词cat的词向量
cat_vec = glove_model['cat']
print(cat_vec)
# 获得单词frog的最相似向量的词汇
print(glove_model.most_similar('frog'))
