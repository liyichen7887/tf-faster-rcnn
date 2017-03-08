from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Set hpyer parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
n_input = 784 
n_classes = 10 
dropout = 0.75 

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create layer models
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# build network
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Prepare model for training
pred = conv_net(x, weights, biases, keep_prob)

# Prepare loss, optimizer, accuracy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# training
with tf.Session() as sess:
    sess.run(init)
    step = 1
    acc = 0
    step_plot = []
    loss_plot = []
    test_plot = []
    #while step * batch_size < training_iters:
    while acc!=1.0:
    	for i in range(78):
        	batch_x, batch_y = mnist.train.next_batch(batch_size)
        	# Run optimization op (backprop)
        	sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
            # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      keep_prob: 1.})
        error = 1 - acc
        step_plot = np.append(step_plot, step)
        loss_plot = np.append(loss_plot, loss)
        test_plot = np.append(test_plot, test_acc)
        print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Error= " + \
                  "{:.5f}".format(error)+ ", Testing Acc= " + \
                  "{:.5f}".format(test_acc))
        step += 1
    print("Optimization Finished!")

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      keep_prob: 1.}))

    plt.plot(step_plot,loss_plot)
    plt.savefig('cnn1.png')
    plt.clf()
    plt.plot(step_plot,test_plot)
    plt.savefig('cnn2.png')