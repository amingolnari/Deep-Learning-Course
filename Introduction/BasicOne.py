"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 3/12/2018

Basic Operations in TensorFlow 
Code 101
"""

import tensorflow as tf

A = tf.constant(12) # A = 12
B = tf.constant(5) # B = 5
C0 = tf.add(A, B) # C0 = A + B
C1 = tf.multiply(A, B) # C1 = A * B

# Some Random Tensors
RndNorm = tf.random_normal(shape = (5, 1), dtype = tf.float32) # Vector with Normal Random Number
RndUn = tf.random_uniform(shape = (10, 3), dtype = tf.float64) # Matrix with Uniform Random Number

tf.div(A, B) # Answer is : 2
tf.truediv(A, B) # Answer is : 2.4

V1 = tf.constant([.3, .25, -.2]) # Shape : (3,)
V2 = tf.constant([.4, .5, -.25])
# Changes the shape by adding -1- to dimensions
V1 = tf.expand_dims(V1, axis = 1) # Shape : (3, 1)
V2 = tf.expand_dims(V2, axis = 1)
MS = tf.reduce_mean(tf.square(V1 - V2)) # Mean Square (V1 - V2)

with tf.Session() as Sess:
	print("C0 : " , Sess.run(C0))
	print("C1 : " , Sess.run(C1))
	print("Normal Random :\n" , Sess.run(RndNorm))
	print("Uniform Random :\n", Sess.run(RndUn))
	print("Mean Square :\n", Sess.run(MS))
