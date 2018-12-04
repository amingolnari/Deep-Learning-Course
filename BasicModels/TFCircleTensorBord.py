"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 4/12/2018

TensorFlow Cicle Classification and Visualiz with TensorBord
Code 204
"""

import numpy as np
import tensorflow as tf
tf.reset_default_graph() # Reset Graph
import matplotlib.pyplot as plt
from sklearn import datasets
# Make Moon Data
moon_X, moon_Y = datasets.make_circles(n_samples = 500,
                                       factor = .2,
                                       noise = .1,
                                       random_state = 0)

# Initialization
InputShape = 2
NumClass = 1
Hidden = 20
lr = 0.5
Epochs = 70
NumShow = 10

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
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                   np.linspace(-1.5, 1.5, 100))
xy = np.c_[x.ravel(), y.ravel()]

# Train Model
with tf.Session() as Sess:
	Sess.run(tf.global_variables_initializer())
	tf.summary.scalar('Loss Function', loss)
	tf.summary.histogram('Prediction', Out)
	FWriter = tf.summary.FileWriter('./log', Sess.graph)
	AllSummary = tf.summary.merge_all()
	for i in range(Epochs):
		_, L, All = Sess.run([Train, loss, AllSummary], {Input: moon_X, Target: np.expand_dims(moon_Y, axis = 1)})
		FWriter.add_summary(All, i)
		if np.mod(i+1, NumShow) == 0: # Plot Each NumShow Step
			print('Epoch %d/%d | Loss : %.4f' % (i+1, Epochs, L))
			Predict = Sess.run(Out, feed_dict = {Input: xy})
			Predict = Predict.reshape(x.shape)
			plt.cla()
			plt.contourf(x, y, Predict, cmap = 'bwr', alpha = 0.2)
			plt.plot(moon_X[moon_Y == 0, 0], moon_X[moon_Y == 0, 1], 'ob')
			plt.plot(moon_X[moon_Y == 1, 0], moon_X[moon_Y == 1, 1], 'or')
			plt.show()
      
"""
Type in your CMD : tensorboard --logdir log/
then click on link --> for axam http://AMIN-PC2:6006 
"""
