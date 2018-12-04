"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
TF Version : 1.12.0
Date : 4/12/2018

TensorFlow CNN Classification on MNIST Data
Code 300
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Load MNIST Data (Download for First)
def ImportData():
	Data = input_data.read_data_sets("/temp/MNIST/", one_hot = True)
	print("# Train data dimension: (%d, %d)" %
	      (Data.train.images.shape[0], Data.train.images.shape[1]))
	print("# Validation data dimension: (%d, %d)" %
	      (Data.validation.images.shape[0], Data.validation.images.shape[1]))
	print("# Test data dimension: (%d, %d)" %
	      (Data.test.images.shape[0], Data.test.images.shape[1]))
	return Data

# Conv2D Layer with Bias and Activation Function
def TFConv2D(Layer, W, b, K = 1):
	Layer = tf.nn.conv2d(Layer, W, strides = [1, K, K, 1], padding = 'SAME')
	Layer = tf.nn.bias_add(Layer, b)
	return tf.nn.relu(Layer)

# MaxPooling2D Layer
def TFMaxPool2D(Layer, K = 2):
	Layer = tf.nn.max_pool(Layer, ksize = [1, K, K, 1], strides = [1, K, K, 1], padding = 'SAME')
	return Layer

# Make CNN Model
def ConvNet(Input, W, b, dropout):
	Input = tf.reshape(Input, shape = [-1, 28, 28, 1]) # Reshape 748 to 28x28 (Images Shape)
	C1 = TFConv2D(Input, W['WC1'], b['bC1'])
	C1 = TFMaxPool2D(C1)
	C2 = TFConv2D(C1, W['WC2'], b['bC2'])
	C2 = TFMaxPool2D(C2)
	F1 = tf.reshape(C2, [-1, W['Wf1'].get_shape().as_list()[0]]) # Flatten Layer
	F1 = tf.add(tf.matmul(F1, W['Wf1']), b['bf1']) # Hidden Layer #1
	F1 = tf.nn.relu(F1)
	F1 = tf.nn.dropout(F1, dropout) # Dropout Layer
	Output = tf.add(tf.matmul(F1, W['Out']), b['Out']) # Output Layer
	return Output

# Initialization Weights and Bias by Random Number
def InitWeightandBias(NumClass):
	W = {
		'WC1': tf.Variable(tf.random_uniform([5, 5, 1, 32])),
		'WC2': tf.Variable(tf.random_uniform([5, 5, 32, 64])),
		'Wf1': tf.Variable(tf.random_uniform([7 * 7 * 64, 256])),
		'Out': tf.Variable(tf.random_uniform([256, NumClass]))
	}
	b = {
		'bC1': tf.Variable(tf.random_uniform([32])),
		'bC2': tf.Variable(tf.random_uniform([64])),
		'bf1': tf.Variable(tf.random_uniform([256])),
		'Out': tf.Variable(tf.random_uniform([NumClass]))
	}
	return W, b

def main():
	Data = ImportData()
	
  # Initialization
	lr = 0.001
	Epochs = 100
	Batch = 256
	NumInput = 784
	NumClass = 10
	Acc = []
	NumBatch = (int)(Data.train.num_examples/Batch) # Total Batches
	
	Input = tf.placeholder(tf.float32, [None, NumInput])
	Label = tf.placeholder(tf.float32, [None, NumClass])
	W, b = InitWeightandBias(NumClass)
	Dropout = tf.placeholder(tf.float32)
	
	Logits = ConvNet(Input, W, b, Dropout)
	Prediction = tf.nn.softmax(Logits)
	Pred = tf.equal(tf.argmax(Prediction, 1), tf.argmax(Label, 1))
	Loss = tf.reduce_mean(tf. nn.softmax_cross_entropy_with_logits(logits = Logits, labels = Label))
	Train = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(Loss)
	Accuracy = tf.reduce_mean(tf.cast(Pred, dtype = tf.float32))
	
	Init = tf.global_variables_initializer()
	
	with tf.Session() as Sess:
		Sess.run(Init)
		for epoch in range(Epochs):
    # Run for All Batches
			for i in range(NumBatch):
				BatchI, BatchL = Data.train.next_batch(Batch)
				Sess.run(Train, feed_dict = {Input: BatchI, Label: BatchL, Dropout: .8})
			acc = Sess.run(Accuracy, feed_dict = {Input: Data.validation.images, Label: Data.validation.labels, Dropout: 1.0})
			Acc.append(acc)
			print('# Epoch %d/%d | Validation Accuracy: %.2f' % ((epoch+1), Epochs, (acc*100)))
		acc = Sess.run(Accuracy, feed_dict = {Input: Data.test.images, Label: Data.test.labels, Dropout: 1.0})
		print('# Test Accuracy: %.2f' % (acc*100))
		plt.figure(1)
		plt.plot(Acc, label = 'Train')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.title('CNN - MNIST')
		plt.grid()
		plt.legend()
		plt.show()
	return

if __name__ == '__main__':
	main()
