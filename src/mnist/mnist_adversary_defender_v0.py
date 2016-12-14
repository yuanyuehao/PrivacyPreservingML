from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1
print_size = batch_size * 256

###################### Define Original Model ############################

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return layer_1, out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
layer1, pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost_pred = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_pred, var_list=[weights['h1'], biases['b1']
    ,weights['h2'], biases['b2'],weights['out'], biases['out']])

###################### Define Attacker Model ############################

n_input_attack = 256 # MNIST data input (img shape: 28*28)
n_classes_attack = 784 # MNIST total classes (0-9 digits)

# tf Graph input
x_attack = tf.placeholder("float", [None, n_classes_attack])

weight_attack = tf.Variable(tf.random_normal([n_input_attack, n_classes_attack]))
bias_attack = tf.Variable(tf.random_normal([n_classes_attack]))

# Construct model
# Specify layer1 as the input of attackers model
x_ = tf.nn.softmax(tf.matmul(layer1, weight_attack) + bias_attack)

#Cost and optimizer
cost_attacker = tf.reduce_mean(tf.square(tf.sub(x_,x_attack) * 10))

optimizer_attacker = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    cost_attacker, var_list=[weight_attack, bias_attack])


###################### Define Defender Model ############################

# Define loss and optimizer
cost_defend = tf.neg(cost_attacker)

optimizer_defend = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_defend, var_list=[weights['h1'], biases['b1']])


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_cost_attacker = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization for original model
            _, c = sess.run([optimizer, cost_pred], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss of original model
            avg_cost += c / total_batch

            # Run optimization for attacker model
            _, c_attacker = sess.run([optimizer_attacker, cost_attacker], feed_dict={x: batch_x, x_attack: batch_x})

            # Run optimization for defender model
            # Comment following line to diable defender 
            # _, _ = sess.run([optimizer_defend, cost_defend], feed_dict={x: batch_x, x_attack: batch_x})

            avg_cost_attacker += c_attacker / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
	    print("Cost of Attacker Model: ", avg_cost_attacker)
            

    print("Optimization Finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


