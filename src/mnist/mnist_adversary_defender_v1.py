
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1
print_size = batch_size * 256

# Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

###################### Define Original Model ############################

# First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# print("h_conv1: ", h_conv1.get_shape())
h_pool1 = max_pool_2x2(h_conv1)    # shape = [batch_size, 14, 14, 32]
# print("h_pool1[0][0]: ", h_pool1[0][0].get_shape())
h_pool1_flat = tf.reshape(h_pool1, [batch_size, -1])

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the origianl Model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


###################### Define Attacker Model ############################

n_input_attack = 14*14*32 
n_classes_attack = 784 

# tf Graph input
x_attack = tf.placeholder("float", [None, n_classes_attack])

weight_attack = tf.Variable(tf.random_normal([n_input_attack, n_classes_attack]))
bias_attack = tf.Variable(tf.random_normal([n_classes_attack]))

# Construct model
# Specify layer1 as the input of attackers model
x_ = tf.nn.softmax(tf.matmul(h_pool1_flat, weight_attack) + bias_attack)

#Cost and optimizer
cost_attacker = tf.reduce_mean(tf.square(tf.sub(x_,x_attack) * 10))

optimizer_attacker = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    cost_attacker, var_list=[weight_attack, bias_attack])

###################### Define Defender Model ############################

# Define loss and optimizer
cost_defend = tf.neg(cost_attacker)

optimizer_defend = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
  cost_defend, var_list=[W_conv1, b_conv1])


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
            # batch = mnist.train.next_batch(batch_size)
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

            # Run optimization for original model
            _, c_attacker = sess.run([optimizer_attacker, cost_attacker], 
              feed_dict={x: batch_x, x_attack: batch_x})

            # Run optimization for defender model
            # Comment following line to diable defender 
            _, _ = sess.run([optimizer_defend, cost_defend], feed_dict={x: batch_x, x_attack: batch_x})

            avg_cost_attacker += c_attacker / total_batch
        
        validation_x = mnist.validation.images
        validation_y = mnist.validation.labels
        # Display logs per epoch step
        # if epoch % display_step == 0:
        #     train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
        #     print "epoch %d, training accuracy %g"%(epoch, train_accuracy)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "epoch %d"%(epoch + 1)
            print "validation accuracy %g test accuracy %g cost of attacker: %g"%(accuracy.eval(feed_dict={x:validation_x, y_: validation_y, keep_prob: 1.0}), 
              accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}), avg_cost_attacker)
            
        
        # print("Cost of Attacker Model: ", avg_cost_attacker)
            
    
    print("Optimization Finished!")
