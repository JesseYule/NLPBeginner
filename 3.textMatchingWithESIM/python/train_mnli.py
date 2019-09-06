import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# parameters主要设置模型相关的参数
# FIXED_PARAMETERS就是一个字典，需要就从里面取出参数
FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

# 选择用什么模型，比如ESIM、biLSTM
model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistently use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


# ————————————————————————————————读取数据————————————————————————————————————

logger.Log("Loading data")

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])

test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])

if 'temp.jsonl' in FIXED_PARAMETERS["test_matched"]:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS["test_matched"])
    logger.Log("Created and removed empty file called temp.jsonl since test set is not available.")

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"


# ————————————————————————————word embedding————————————————————————————————————
if not os.path.isfile(dictpath):
    logger.Log("Building dictionary")
    if FIXED_PARAMETERS["alpha"] == 0:
        word_indices = build_dictionary([training_mnli])
    else:
        word_indices = build_dictionary([training_mnli])
    
    logger.Log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [training_mnli,
                                        test_matched, test_mismatched])
    pickle.dump(word_indices, open(dictpath, "wb"))
else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))  # word_indices是一个词典，每个单词对应一个编号，编号总长度就是单词种类的数量
    logger.Log("Padding and indexifying sentences")
    #  将句子转换成向量表示
    sentences_to_padded_index_sequences(word_indices, [training_mnli,
                                        test_matched, test_mismatched])

logger.Log("Loading embeddings")
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


# ——————————————————————————————创建模型——————————————————————————————————————————
class modelClassifier:
    def __init__(self, seq_length):
        # Define hyperparameters
        self.learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 5
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]

        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim, 
                                hidden_dim=self.dim, embeddings=loaded_embeddings, 
                                emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # Boolean stating that training has not been completed, 
        self.completed = False 

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres

    def classify(self, examples):
        # This classifies a list of examples
        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        genres = []
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_minibatch(examples, 
                                    self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), cost

    def train(self, train_mnli):
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 0
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_mtrain_acc = 0.
        self.last_train_acc = [.001, .001, .001]
        self.best_step = 0

        # Training cycle
        logger.Log("Training...")

        while True:
            training_data = train_mnli
            random.shuffle(training_data)  # 随机重排训练数据
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)

            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the
                # cost function for logging
                # 上面通过minibatch函数从训练集中随机抽取数据，这里填入模型中
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                             self.model.hypothesis_x: minibatch_hypothesis_vectors,
                             self.model.y: minibatch_labels,
                             self.model.keep_rate_ph: self.keep_rate}

                # 正式训练，计算损失
                _, c = self.sess.run([self.optimizer, self.model.total_cost], feed_dict)

                # Since a single epoch can take a  ages for larger models (ESIM),
                # we'll print  accuracy every 50 steps
                # 这里的意思是每训练了50步，就检验一下当前的模型
                # 从代码可以看出主要是用train_mnli的前五千条数据放进模型里面训练，分析模型输出和实际结果的差异
                if self.step % self.display_step_freq == 0:

                    mtrain_acc, mtrain_cost = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)

                    logger.Log("Step: %i\t MultiNLI train acc: %f" % (self.step, mtrain_acc))
                    logger.Log("Step: %i\t MultiNLI train cost: %f" % (self.step, mtrain_cost))

                self.step += 1

                # Compute average loss
                # 每次训练都返回一次损失c，所以计算平均损失就相加再除以总训练次数
                avg_cost += c / (total_batch * self.batch_size)

            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" % (self.epoch + 1, avg_cost))

            self.epoch += 1
            self.last_train_acc[(self.epoch % 5) - 1] = mtrain_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc) / (5 * min(self.last_train_acc)) - 1)

            # 训练次数超过30000次就停止训练
            if (progress < 0.1) or (self.step > self.best_step + 30000):
                logger.Log("MultiNLI Train accuracy: %s" % (self.best_mtrain_acc))
                self.completed = True
                break


classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

classifier.train(training_mnli)

logger.Log("Acc on matched multiNLI dev-set: %s"
    % (evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
logger.Log("Acc on mismatched multiNLI dev-set: %s"
    % (evaluate_classifier(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
