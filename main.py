"""
	Author: Arsh Zahed
	Date: 8-21-2017

	For the blog post found on www.callaunchpad.org

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Training Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#Parameters
learning_rate = 0.01
num_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

#Network Parameters
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

#Visible Layer
X = tf.placeholder("float", [None, n_input])

"""
Note how the hidden layers of the encoder and decorder are "mirrored".
"""
weights = {
	'e_1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'e_2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'd_1' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'd_2' : tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
	'e_1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'e_2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'd_1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'd_2' : tf.Variable(tf.random_normal([n_input]))
}

def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['e_1']),
                                   biases['e_1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['e_2']),
                                   biases['e_2']))
    return layer_2

def decoder(x):
	# Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['d_1']),
                                   biases['d_1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['d_2']),
                                   biases['d_2']))
    return layer_2

#Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#Prediction
y_pred = decoder_op

#Targets (Labels) are the input data
y_true = X

#Define loss and optimizer, minimal square error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

#Launch the graph
#Launch the graph
with tf.Session() as sess:
    sess.run(init)	
    total_batch = int(mnist.train.num_examples/batch_size)

    #Train
    for epoch in range(num_epochs):
        #Loop over batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    #Applying encode and decode overt test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
    plt.draw()










