from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tensorflow as tf

url = 'http://cogcomp.org/Data/QA/QC/'
dir_name = 'data/question-classif-data'


#  下载数据
def maybe_download(dir_name, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if not os.path.exists(os.path.join(dir_name, filename)):
        filename, _ = urlretrieve(os.path.join(dir_name, filename))

    print(os.path.join(dir_name, filename))
    statinfo = os.stat(os.path.join(dir_name, filename))
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % os.path.join(dir_name, filename))
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + os.path.join(dir_name, filename) + '. Can you get to it with a browser?')
    return filename


# 读取数据
def read_data(filename):
    '''
    Read data from a file with given filename
    Returns a list of strings where each string is a lower case word
    '''
    global max_sent_length  # 最大句子长度：33
    questions = []
    labels = []
    with open(filename, 'r', encoding='latin-1') as f:
        for row in f:
            row_str = row.split(":")
            lb, q = row_str[0], row_str[1]
            q = q.lower()
            labels.append(lb)
            questions.append(q.split())
            if len(questions[-1]) > max_sent_length:  # 检测每个句子最大的长度
                max_sent_length = len(questions[-1])
    return questions, labels


# 这里就是词袋模型，把文本转化为词向量
def build_dataset(questions):
    words = []
    data_list = []
    count = []

    # First create a large list with all the words in all the questions
    for d in questions:
        words.extend(d)
    print('%d Words found.' % len(words))
    print('Found %d words in the vocabulary. ' % len(collections.Counter(words).most_common()))

    # Sort words by there frequency
    count.extend(collections.Counter(words).most_common())

    # Create an ID for each word by giving the current length of the dictionary
    # And adding that item to the dictionary
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # Traverse through all the text and
    # replace the string words with the ID
    # of the word found at that index
    for d in questions:
        data = list()
        for word in d:
            index = dictionary[word]
            data.append(index)

        data_list.append(data)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data_list, count, dictionary, reverse_dictionary


class BatchGenerator(object):
    '''
    Generates a batch of data
    '''

    def __init__(self, batch_size, questions, labels):
        self.questions = questions
        self.labels = labels
        self.text_size = len(questions)
        self.batch_size = batch_size
        self.data_index = 0
        assert len(self.questions) == len(self.labels)

    def generate_batch(self):
        '''
        Data generation function. This outputs two matrices
        inputs: a batch of questions where each question is a tensor of size
        [sent_length, vocabulary_size] with each word one-hot-encoded
        labels_ohe: one-hot-encoded labels corresponding to the questions in inputs
        '''
        global sent_length, num_classes
        global dictionary, all_labels

        # Numpy arrays holding input and label data
        # 输入就是一个batch的句子，所以是一个batch*最大句子长度*单词长度
        inputs = np.zeros((self.batch_size, sent_length, vocabulary_size), dtype=np.float32)
        labels_ohe = np.zeros((self.batch_size, num_classes), dtype=np.float32)

        # When we reach the end of the dataset
        # start from beginning
        if self.data_index + self.batch_size >= self.text_size:
            self.data_index = 0

        # For each question in the dataset
        for qi, que in enumerate(self.questions[self.data_index:self.data_index + self.batch_size]):
            # For each word in the question
            for wi, word in enumerate(que):
                # Set the element at the word ID index to 1
                # this gives the one-hot-encoded vector of that word
                inputs[qi, wi, dictionary[word]] = 1.0

            # Set the index corrsponding to that particular class to 1
            labels_ohe[qi, all_labels.index(self.labels[self.data_index + qi])] = 1.0

        # Update the data index to get the next batch of data
        self.data_index = (self.data_index + self.batch_size) % self.text_size

        return inputs, labels_ohe

    def return_index(self):
        # Get the current index of data
        return self.data_index


if __name__ == '__main__':

    filename = maybe_download(dir_name, 'train_1000.label', 60774)
    test_filename = maybe_download(dir_name, 'TREC_10.label', 23354)

    filenames = ['train_1000.label', 'TREC_10.label']
    num_files = len(filenames)
    for i in range(len(filenames)):
        file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
        assert file_exists
    print('Files found and verified.')

    max_sent_length = 0

    # Process train and Test data
    for i in range(num_files):
        print('\nProcessing file %s' % os.path.join(dir_name, filenames[i]))
        if i == 0:
            # Processing training data
            train_questions, train_labels = read_data(os.path.join(dir_name, filenames[i]))
            # Making sure we got all the questions and corresponding labels
            assert len(train_questions) == len(train_labels)
        elif i == 1:
            # Processing testing data
            test_questions, test_labels = read_data(os.path.join(dir_name, filenames[i]))
            # Making sure we got all the questions and corresponding labels.
            assert len(test_questions) == len(test_labels)

        # Print some data to see everything is okey
        for j in range(5):
            print('\tQuestion %d: %s' % (j, train_questions[j]))
            print('\tLabel %d: %s\n' % (j, train_labels[j]))

    print('Max Sentence Length: %d' % max_sent_length)
    print('\nNormalizing all sentences to same length')

    # 因为CNN不像RNN，需要确保每次输入（每个句子）的维度相等，也就是每次输入都是同样尺寸的二维矩阵
    # 填充每个句子，使每个句子长度相等
    for qi, que in enumerate(train_questions):
        for _ in range(max_sent_length - len(que)):
            que.append('PAD')
        assert len(que) == max_sent_length
        train_questions[qi] = que
    print('Train questions padded')

    # 填充每个句子，使每个句子长度相等
    for qi, que in enumerate(test_questions):
        for _ in range(max_sent_length - len(que)):
            que.append('PAD')
        assert len(que) == max_sent_length
        test_questions[qi] = que
    print('\nTest questions padded')

    # Printing a test question to see if everything is correct
    print('\nSample test question: %s', test_questions[0])

    # Create a dataset with both train and test questions
    all_questions = list(train_questions)
    all_questions.extend(test_questions)

    # Use the above created dataset to build the vocabulary
    all_question_ind, count, dictionary, reverse_dictionary = build_dataset(all_questions)

    # Print some statistics about the processed data
    print('All words (count)', count[:5])
    print('\n0th entry in dictionary: %s', reverse_dictionary[0])
    print('\nSample data', all_question_ind[0])
    print('\nSample data', all_question_ind[1])
    print('\nVocabulary: ', len(dictionary))
    vocabulary_size = len(dictionary)

    print('\nNumber of training questions: ', len(train_questions))
    print('Number of testing questions: ', len(test_questions))

    batch_size = 16  # We process 16 questions at a time
    sent_length = max_sent_length

    num_classes = 6  # Number of classes
    # All the types of question that are in the dataset
    all_labels = ['NUM', 'LOC', 'HUM', 'DESC', 'ENTY', 'ABBR']

    # Test our batch generator
    sample_gen = BatchGenerator(batch_size, train_questions, train_labels)
    # Generate a single batch
    sample_batch_inputs, sample_batch_labels = sample_gen.generate_batch()
    # Generate another batch
    sample_batch_inputs_2, sample_batch_labels_2 = sample_gen.generate_batch()

    # Make sure that we infact have the question 0 as the 0th element of our batch
    assert np.all(np.asarray([dictionary[w] for w in train_questions[0]], dtype=np.int32)
                  == np.argmax(sample_batch_inputs[0, :, :], axis=1))

    # Print some data labels we obtained
    print('Sample batch labels')
    print(np.argmax(sample_batch_labels, axis=1))
    print(np.argmax(sample_batch_labels_2, axis=1))

    tf.reset_default_graph()

    batch_size = 32
    # Different filter sizes we use in a single convolution layer
    filter_sizes = [3, 5, 7]

    # inputs and labels
    sent_inputs = tf.placeholder(shape=[batch_size, sent_length, vocabulary_size], dtype=tf.float32,
                                 name='sentence_inputs')
    sent_labels = tf.placeholder(shape=[batch_size, num_classes], dtype=tf.float32, name='sentence_labels')

    # 3 filters with different context window sizes (3,5,7)
    # Each of this filter spans the full one-hot-encoded length of each word and the context window width

    # Weights of the first parallel layer
    w1 = tf.Variable(tf.truncated_normal([filter_sizes[0], vocabulary_size, 1], stddev=0.02, dtype=tf.float32),
                     name='weights_1')
    b1 = tf.Variable(tf.random_uniform([1], 0, 0.01, dtype=tf.float32), name='bias_1')

    # Weights of the second parallel layer
    w2 = tf.Variable(tf.truncated_normal([filter_sizes[1], vocabulary_size, 1], stddev=0.02, dtype=tf.float32),
                     name='weights_2')
    b2 = tf.Variable(tf.random_uniform([1], 0, 0.01, dtype=tf.float32), name='bias_2')

    # Weights of the third parallel layer
    w3 = tf.Variable(tf.truncated_normal([filter_sizes[2], vocabulary_size, 1], stddev=0.02, dtype=tf.float32),
                     name='weights_3')
    b3 = tf.Variable(tf.random_uniform([1], 0, 0.01, dtype=tf.float32), name='bias_3')

    # Fully connected layer
    w_fc1 = tf.Variable(tf.truncated_normal([len(filter_sizes), num_classes], stddev=0.5, dtype=tf.float32),
                        name='weights_fulcon_1')
    b_fc1 = tf.Variable(tf.random_uniform([num_classes], 0, 0.01, dtype=tf.float32), name='bias_fulcon_1')

    # Calculate the output for all the filters with a stride 1
    # We use relu activation as the activation function
    h1_1 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w1, stride=1, padding='SAME') + b1)
    h1_2 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w2, stride=1, padding='SAME') + b2)
    h1_3 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w3, stride=1, padding='SAME') + b3)

    # Pooling over time operation

    # This is doing the max pooling. Thereare two options to do the max pooling
    # 1. Use tf.nn.max_pool operation on a tensor made by concatenating h1_1,h1_2,h1_3 and converting that tensor to 4D
    # (Because max_pool takes a tensor of rank >= 4 )
    # 2. Do the max pooling separately for each filter output and combine them using tf.concat
    # (this is the one used in the code)

    h2_1 = tf.reduce_max(h1_1, axis=1)
    h2_2 = tf.reduce_max(h1_2, axis=1)
    h2_3 = tf.reduce_max(h1_3, axis=1)

    h2 = tf.concat([h2_1, h2_2, h2_3], axis=1)

    # Calculate the fully connected layer output (no activation)
    # Note: since h2 is 2d [batch_size,number of parallel filters]
    # reshaping the output is not required as it usually do in CNNs
    logits = tf.matmul(h2, w_fc1) + b_fc1

    # Loss (Cross-Entropy)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=sent_labels, logits=logits))

    # Momentum Optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

    # With filter widths [3,5,7] and batch_size 32 the algorithm
    # achieves around ~90% accuracy on test dataset (50 epochs).
    # From batch sizes [16,32,64] I found 32 to give best performance

    session = tf.InteractiveSession()

    num_steps = 50  # Number of epochs the algorithm runs for

    # Initialize all variables
    tf.global_variables_initializer().run()
    print('Initialized\n')

    # Define data batch generators for train and test data
    train_gen = BatchGenerator(batch_size, train_questions, train_labels)
    test_gen = BatchGenerator(batch_size, test_questions, test_labels)

    # How often do we compute the test accuracy
    test_interval = 1

    # Compute accuracy for a given set of predictions and labels
    def accuracy(labels, preds):
        return np.sum(np.argmax(labels, axis=1) == preds) / labels.shape[0]


    # Running the algorithm
    for step in range(num_steps):
        avg_loss = []

        # A single traverse through the whole training set
        for tr_i in range((len(train_questions) // batch_size) - 1):
            # Get a batch of data
            tr_inputs, tr_labels = train_gen.generate_batch()
            # Optimize the network and compute the loss
            l, _ = session.run([loss, optimizer], feed_dict={sent_inputs: tr_inputs, sent_labels: tr_labels})
            avg_loss.append(l)

        # Print average loss
        print('Train Loss at Epoch %d: %.2f' % (step, np.mean(avg_loss)))
        test_accuracy = []

        # Compute the test accuracy
        if (step + 1) % test_interval == 0:
            for ts_i in range((len(test_questions) - 1) // batch_size):
                # Get a batch of test data
                ts_inputs, ts_labels = test_gen.generate_batch()
                # Get predictions for that batch
                preds = session.run(predictions, feed_dict={sent_inputs: ts_inputs, sent_labels: ts_labels})
                # Compute test accuracy
                test_accuracy.append(accuracy(ts_labels, preds))

            # Display the mean test accuracy
            print('Test accuracy at Epoch %d: %.3f' % (step, np.mean(test_accuracy) * 100.0))


