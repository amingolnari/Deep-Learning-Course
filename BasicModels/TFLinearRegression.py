"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 4/12/2018

TensorFlow Linear Regression
Code 200
"""

import numpy as np
import tensorflow as tf
tf.reset_default_graph() # Reset Graph
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100, dtype = 'float32') # x = 0 --> 2*pi with lenght 100
y = -x + 2*np.random.rand(x.shape[0]) # y = -x + noise

# Expand Dim (n,) --> (n, 1)
x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

# Initialization
InputShape = 1
NumClass = 1
lr = .01
Epochs = 50

# TF Input Graph and Target
Input = tf.placeholder(tf.float32, [None, InputShape])
Target = tf.placeholder(tf.float32, [None, NumClass])

# Weight and Bias
Weight = tf.Variable(tf.zeros([InputShape, NumClass]))
Bias = tf.Variable(tf.ones([NumClass]))

# TF Output Graph
Out = tf.add(tf.matmul(Input, Weight) ,Bias)

# Loss Function and Minimization MSE (Target - Output)
loss = tf.losses.mean_squared_error(Target, Out)
Train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Init Graph
Sess = tf.Session()
Sess.run(tf.global_variables_initializer())

# Train The Model
for i in range(Epochs):
	Sess.run(Train, feed_dict = {Input: x, Target: y})

# Plot Regression
Predict = Sess.run(Out, {Input: x, Target: y})
plt.plot(x, Predict, 'b')
plt.plot(x, y, 'r.')
plt.show()
