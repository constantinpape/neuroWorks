# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
    dataset = dataset.reshape(
      (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def vanilla_cnn(num_steps = 1001, verbose = False):

    patch_size = 5
    depth = 16
    num_hidden = 64
    batch_size = 16

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        x = tf.placeholder(
          tf.float32, shape=(None, image_size, image_size, num_channels))
        y = tf.placeholder(tf.float32, shape=(None, num_labels))

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(x)
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, y))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(logits)

        accuracy = 100. * tf.reduce_mean( tf.cast(
            tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)),
            tf.float32 ))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            _, l, predictions = session.run(
            [optimizer, loss, prediction], feed_dict={x:batch_data,y:batch_labels})
            if (step % 50 == 0 and verbose):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:batch_data,y:batch_labels}))
                print('Validation accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:valid_dataset,y:valid_labels}))

        print('Test accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:test_dataset,y:test_labels}))


def mpool_cnn(num_steps = 1001, verbose = False):

    patch_size = 5
    depth = 16
    num_hidden = 64
    batch_size = 16

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        x = tf.placeholder(
          tf.float32, shape=(None, image_size, image_size, num_channels))
        y = tf.placeholder(tf.float32, shape=(None, num_labels))

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):

            # conv layer 1
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            mpool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')

            # conv layer 2
            conv = tf.nn.conv2d(mpool, layer2_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            mpool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')

            # fully connected layer
            shape = mpool.get_shape().as_list()
            reshape = tf.reshape(mpool, [-1, shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(x)
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, y))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(logits)

        accuracy = 100. * tf.reduce_mean( tf.cast(
            tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)),
            tf.float32 ))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            _, l, predictions = session.run(
            [optimizer, loss, prediction], feed_dict={x:batch_data,y:batch_labels})
            if (step % 50 == 0 and verbose):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:batch_data,y:batch_labels}))
                print('Validation accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:valid_dataset,y:valid_labels}))

        print('Test accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:test_dataset,y:test_labels}))


def best_cnn(num_steps = 3001, verbose = False):

    patch_size = 5
    depth = 16
    batch_size = 16

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        x = tf.placeholder(
          tf.float32, shape=(None, image_size, image_size, num_channels))
        y = tf.placeholder(tf.float32, shape=(None, num_labels))

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(.1, global_step, 1000, .9)

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, 6], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([6]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, 6, 16], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * 16, 120], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[120]))

        layer4_weights = tf.Variable(tf.truncated_normal(
            [120, 84], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[84]))

        layer5_weights = tf.Variable(tf.truncated_normal(
            [84, num_labels], stddev=0.1))
        layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):

            # conv layer 1
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            mpool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')

            # conv layer 2
            conv = tf.nn.conv2d(mpool, layer2_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            mpool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')

            # fully connected layer 1
            shape = mpool.get_shape().as_list()
            reshape = tf.reshape(mpool, [-1, shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

            # fully connected layer 2
            hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)

            return tf.matmul(hidden, layer5_weights) + layer5_biases

        # Training computation.
        logits = model(x)
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, y))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(logits)

        accuracy = 100. * tf.reduce_mean( tf.cast(
            tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)),
            tf.float32 ))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            _, l, predictions = session.run(
            [optimizer, loss, prediction], feed_dict={x:batch_data,y:batch_labels})
            if (step % 50 == 0 and verbose):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:batch_data,y:batch_labels}))
                print('Validation accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:valid_dataset,y:valid_labels}))

        print('Test accuracy: %.1f%%' % session.run(accuracy, feed_dict={x:test_dataset,y:test_labels}))



def run_vanilla():
    vanilla_cnn(verbose = True)


def ex1():
    mpool_cnn(verbose = True)


def ex2():
    best_cnn(verbose = True)


# Ex 1: Replace stride 2 convolutions with max pooling of stride 2 and size 2
# ( tf.nn.max_pool() )

# Ex 2: Best performance of a convnet via Architecture (cf. LeNet5), Dropout, learning rate decay


if __name__ == '__main__':
    #run_vanilla()
    ex2()


# Results:

# Vanilla CNN: 88.2 %
# Ex 1: 88.7 %
# Ex 2:
# Adding weight decay: 93.4 %
# LeNet Architecture: 93.1 %
