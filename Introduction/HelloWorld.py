"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 3/12/2018

TensorFlow Hello World :) 
Code 100
"""

import tensorflow as tf

# Constant String
Str = tf.constant('Hello World!')

Sess = tf.Session()
print(Sess.run(Str)) # Output : b'Hello World!'
print(Sess.run(Str).decode()) # Output : Hello World!
