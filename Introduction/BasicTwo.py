"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 3/12/2018

Basic Operations in TensorFlow 
Code 102
"""

import tensorflow as tf
"""
Variable : https://www.tensorflow.org/api_docs/python/tf/Variable
Plaseholder : https://www.tensorflow.org/api_docs/python/tf/placeholder

tf.Variable used for Weights and Bias (Trainable Parameter)
tf.placeholder used for Data (non Trainable Parameter)
"""

# Initialization (for example)
NumClass = 10
InputShape = 100

# TensorFlow Graph Input
Input = tf.placeholder(tf.float32, [None, InputShape]) # Input Data
Weight = tf.Variable(tf.random_uniform([InputShape, NumClass]))
Bias = tf.Variable(tf.zeros([NumClass]))

# TensorFlow Graph Output
Output = tf.add(tf.matmul(Input, Weight), Bias) # Linear Function
