"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
Keras Version : 2.2.4
Date : 4/12/2018

Keras Circle Classification and Visualize with TensorBoard
Code 205
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.callbacks import TensorBoard

# Method for Make Keras Model with one Hidden Layer
def KerasModel(Hidden, InputShape, NumClass, lr):
	model = Sequential()
	model.add(Dense(Hidden, input_shape = [InputShape,],
	                activation = 'tanh',
                    use_bias = True))	model.add(Dense(NumClass,
                    activation = 'sigmoid',
                    use_bias = True))
  # Adam Optimizer with Binary Cross Entropy Loss Function
	model.compile(Adam(lr = lr),
                  loss =  'binary_crossentropy',
                  metrics=['accuracy'])
	return model

def PlotClassification(model, moon_X, moon_Y):
	# For Contour Prediction
	x, y = np.meshgrid(np.linspace(-1.5, 1.5, 200),
	                   np.linspace(-1.5, 1.5, 200))
	xy = np.c_[x.ravel(), y.ravel()]
	Predict = model.predict(xy)
	Predict = Predict.reshape(x.shape)
	plt.cla()
	plt.contourf(x, y, Predict, cmap = 'bwr', alpha = 0.2)
	plt.plot(moon_X[moon_Y == 0, 0], moon_X[moon_Y == 0, 1], 'ob')
	plt.plot(moon_X[moon_Y == 1, 0], moon_X[moon_Y == 1, 1], 'or')
	plt.show()

def main():
	# Make Moon Data
	moon_X, moon_Y = \
		datasets.make_circles(
			n_samples = 500,
			factor = .2,
			noise = .1,
			random_state = 0)
	
	# Initialization
	InputShape = 2
	NumClass = 1
	Hidden = 20
	lr = 0.5
	Epochs = 7
	Batch = 10
  
  # Make Keras Model
	model = KerasModel(Hidden, # Number of Hidden Neuron
	                   InputShape, #
	                   NumClass,
	                   lr)
  # TensorBoard Callback - save file in log folder
	TBCal = TensorBoard(log_dir = 'log/',
	                    histogram_freq = 1,
	                    write_grads = True)
	model.fit(moon_X, moon_Y,
	          epochs = Epochs,
	          batch_size = Batch,
	          verbose = 2, # Show result method
            validation_split = .2, # Set 20% for Validation Check
	          callbacks = [TBCal])
	PlotClassification(model, moon_X, moon_Y)
	
if __name__ == '__main__':
	main()
  

"""
Type in your CMD : tensorboard --logdir log/
then click on link --> for axam http://AMIN-PC2:6006 
"""
