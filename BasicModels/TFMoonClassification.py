"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 4/12/2018

TensorFlow Moon Classification
Code 203
"""

import numpy as np
import tensorflow as tf
tf.reset_default_graph() # Reset Graph
import matplotlib.pyplot as plt
from sklearn import datasets

# Make Moon Data
moon_X, moon_Y = datasets.make_moons(n_samples = 200,
                                     noise = .2,
                                     random_state = 0)

# Initialization
InputShape = 2
NumClass = 1
Hidden = 10
lr = 0.2
Epochs = 100
NumShow = 20

# TF Input Graph and Target
Input = tf.placeholder(tf.float32, [None, InputShape])
Target = tf.placeholder(tf.float32, [None, NumClass])

# Hidden Layer #1
Hidden1 = tf.layers.dense(inputs = Input,
                          units = Hidden,
                          activation = tf.nn.tanh)
# TF Output Graph
Out = tf.layers.dense(inputs = Hidden1,
                      units = NumClass,
                      activation = tf.nn.sigmoid)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Out, labels = Target))
Train = tf.train.AdamOptimizer(lr).minimize(loss)

# For Contour Prediction
x, y = np.meshgrid(np.linspace(-1.5, 2.5, 100),
                   np.linspace(-1.5, 2.0, 100))
xy = np.c_[x.ravel(), y.ravel()]

# Train Model
with tf.Session() as Sess:
	Sess.run(tf.global_variables_initializer())
	for i in range(Epochs):
		_, L = Sess.run([Train, loss], {Input: moon_X, Target: np.expand_dims(moon_Y, axis = 1)})
		if np.mod(i+1, NumShow) == 0: # Plot Each NumShow Step
			print('Epoch %d/%d | Loss : %.4f' % (i+1, Epochs, L))
			Predict = Sess.run(Out, feed_dict = {Input: xy})
			Predict = Predict.reshape(x.shape)
			plt.cla()
			plt.contourf(x, y, Predict, cmap = 'bwr', alpha = 0.2)
			plt.plot(moon_X[moon_Y == 0, 0], moon_X[moon_Y == 0, 1], 'ob')
			plt.plot(moon_X[moon_Y == 1, 0], moon_X[moon_Y == 1, 1], 'or')
			plt.show()
