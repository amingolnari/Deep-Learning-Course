"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 4/12/2018

TensorFlow nonLinear Regression
Code 201
"""

import numpy as np
import tensorflow as tf
tf.reset_default_graph() # Reset Graph
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100, dtype = 'float32') # x = 0 --> 2*pi with lenght 100
y = (x**2) - (3*x) + (3*np.random.rand(x.shape[0])) # y = x^2 - 3x + noise

# Expand Dim (n,) --> (n, 1)
x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

# Initialization
InputShape = x.shape
NumClass = y.shape
Hidden = 50
lr = .6
Epochs = 200
NumShow = 10

# TF Input Graph and Target
Input = tf.placeholder(tf.float32, InputShape)
Target = tf.placeholder(tf.float32, NumClass)

# Hidden Layer #1 
Hidden1 = tf.layers.dense(inputs = Input,
                          units = Hidden,
                          activation = tf.nn.tanh)
# TF Output Graph
Out = tf.layers.dense(inputs = Hidden1,
                      units = 1)

# Loss Function and Minimization MSE (Target - Output)
loss = tf.losses.mean_squared_error(Target, Out)
Train = tf.train.AdamOptimizer(lr).minimize(loss)

# Init Graph
Sess = tf.Session()
Sess.run(tf.global_variables_initializer())

# Train The Model
for i in range(Epochs):
	_, L = Sess.run([Train, loss], feed_dict = {Input: x, Target: y})
	if np.mod(i+1, NumShow) == 0: # Show Result
		print('Epoch %d/%d | Loss : %.4f' % (i+1, Epochs, L))
		Predict = Sess.run(Out, {Input: x, Target: y})
		plt.cla()
		plt.plot(x, Predict, 'b')
		plt.plot(x, y, 'r.')
		plt.show()
