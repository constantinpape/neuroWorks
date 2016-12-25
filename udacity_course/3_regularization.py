# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

image_size = 28
num_labels = 10

def load_data():

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

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def log_reg(beta, verbose = False, num_steps = 3001):

    batch_size = 128
    graph = tf.Graph()

    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
          tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta * tf.nn.l2_loss(weights)

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
          tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0 and verbose):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                  valid_prediction.eval(), valid_labels))

        print("Run with beta = %f" % (beta,))
        # FIXME this does not work for some reason
        #print("Train accuracy: %.1f%%" % accuracy(train_prediction.eval(), train_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def two_layer_nn(beta, verbose = False, num_steps = 3001, dropout = 1.):

    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        x = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
        y = tf.placeholder(tf.float32, shape=(None, num_labels))
        # Fixed validation and training sets
        keep_prob = tf.placeholder(tf.float32)

        # Variables.
        n_hidden = 1024

        # first layer
        w1 = tf.Variable(
          tf.truncated_normal([image_size * image_size, n_hidden]))
        b1 = tf.Variable(tf.zeros([n_hidden]))

        # second layer
        w2 = tf.Variable(
          tf.truncated_normal([n_hidden, num_labels]))
        b2 = tf.Variable(tf.zeros([num_labels]))

        l1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
        l1 = tf.nn.dropout(l1, keep_prob)
        l2 = tf.matmul( l1, w2) + b2

        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(l2, y)) + beta * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Evaluation
        out = tf.nn.softmax(l2)
        tf_accuracy = 100. * tf.reduce_mean( tf.cast(
            tf.equal(tf.argmax(out,1), tf.argmax(y,1)),
            tf.float32 ))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            # run optimisation with dropout
            session.run(optimizer, feed_dict = {x : batch_data,
                                    y : batch_labels,
                                    keep_prob : dropout} )

            l, predictions = session.run(
                [loss, out], feed_dict = {x : batch_data,
                                         y : batch_labels,
                                         keep_prob : 1.} )

            if (step % 500 == 0 and verbose):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % session.run(
                    tf_accuracy, feed_dict={x : valid_dataset,
                                            y : valid_labels,
                                            keep_prob : 1.}))

        print("Run with beta = %f" % (beta,))
        print("Train accuracy: %.1f%%" % session.run(
            tf_accuracy, feed_dict={x : train_dataset,
                                    y : train_labels,
                                    keep_prob : 1.}))
        print("Test accuracy: %.1f%%" % session.run(
            tf_accuracy, feed_dict={x : test_dataset,
                                    y : test_labels,
                                    keep_prob : 1.}))


# Ex 1: NN and log reg with l2 norm

def ex1():

    for beta in (1e-3,5*1e-3,1e-2,5*1e-2):
        log_reg(beta)

    for beta in (1e-4,5*1e-4,1e-3,5*1e-3,1e-2,5*1e-2):
        two_layer_nn(beta)


# Ex 2: Use just a few training batches.
# To show effects of overfitting
# I don't fully get this exercise...
# As we are only seeing a fraction of the training data, we
# can't overfit to the whole training set
# probably we'd see overfitting if we monitor the training error for the batches during training

def ex2():

    beta = 1e-3
    n_batches = 25

    log_reg(beta, num_steps = n_batches)
    two_layer_nn(beta, num_steps = n_batches)


# Ex 3: dropout

def ex3():

    beta = 1e-3
    two_layer_nn(beta, dropout = .8)
    two_layer_nn(beta, dropout = .8, num_steps = 25)




if __name__ == '__main__':
    ex3()


# Results

# Ex 1:
# LogReg:
# beta = .001: 88.8 %
# beta = .005: 88.9 %
# beta = .01:  88.6 %
# beta = .05:  87.1 %
# TwoLayerNN:
# beta = .0001: 89.2 %
# beta = .0005: 91.0 %
# beta = .001:  92.8 %
# beta = .005:  91.5 %
# beta = .01:   90.2 %
# beta = .05:   85.3 %

# Ex 2:
# beta = 1e-3, 25 batches
# LogReg
# acc_train = - TODO fix train accuracy
# acc_test  = 63.0 %
# TwoLayerNN
# acc_train = -
# acc_test  = 84.0 %

# Ex 3:
# beta = 1e-3
# Dropout .8 full batches
# 92.6 %
# only 25 batches
# 82.5 %
