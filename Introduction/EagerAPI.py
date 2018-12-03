"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 3/12/2018

TensorFlow's Eager API
Code 103
"""

import numpy as np
import tensorflow as tf
# Enable Eager API
tf.enable_eager_execution()

A = tf.constant([[.25, 5.6], 
                 [2.0, -6.5]], dtype = tf.float32) # Tensor with shape (2, 2)
B = np.array([[1.5, -2.1], 
              [-1.5, 2.1]], dtype = 'float32') # Numpy Array with shape (2, 2)

C0 = A * B
C1 = tf.multiply(A, B)

print(C0, '\n', C1)
print(C0[0][0])
print(C1[0][0])
