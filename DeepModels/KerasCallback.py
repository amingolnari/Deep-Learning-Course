"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
Keras Version : 2.2.4
Date : 8/12/2018

Using Callback in Keras
Code 302
"""

"""
Stop Training with validation loss Check
Change Learning Rate Per n Epoch
Save Learning Rate Per Train Epoch
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.callbacks import TensorBoard, Callback
from keras import backend as K

# Method for Make Keras Model with one Hidden Layer
def KerasModel(Hidden, InputShape, NumClass, lr):
	model = Sequential()
	model.add(Dense(Hidden, input_shape = [InputShape, ],
	                activation = 'tanh',
	                use_bias = True))
	model.add(Dense(NumClass,
	                activation = 'sigmoid',
	                use_bias = True))
	# Adam Optimizer with Binary Cross Entropy Loss Function
	model.compile(Adam(lr = lr),
	              loss = 'binary_crossentropy',
	              metrics = ['accuracy'])
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
	global MinLoss, ChangePer
	
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
	Epochs = 100
	Batch = 128
	MinLoss = 3.5e-3 # 0.0035
	ChangePer = 5
	
	# Make Keras Model
	model = KerasModel(Hidden,  # Number of Hidden Neuron
	                   InputShape,  #
	                   NumClass,
	                   lr)
		
	# Set Callback to use in model.fit
	call = Call()

	# Train Model
  model.fit(moon_X, moon_Y, # Inputs
	          epochs = Epochs,
	          batch_size = Batch,
	          verbose = 0,  # Show result method
	          validation_split = .2,  # Set 20% Train Data for Validation Check
	          callbacks = [call])
	
	# Plot Result
	PlotClassification(model, moon_X, moon_Y)
  
  # Plot Learning Rate in Train Epochs
	plt.plot(call.learninrate, 'ro')
	plt.grid()
	plt.xlabel('Epochs')
	plt.ylabel('Learning Rate')
	plt.xlim(-0.1, len(call.learninrate)+.1)
	plt.show()

class Call(Callback):
	def on_train_begin(self, logs=None):
		self.learninrate = []
		
	def on_epoch_begin(self, epoch, logs=None):
		# Append Learning Rate to learninrate
    self.learninrate.append(K.get_value(self.model.optimizer.lr))
		 
	def on_epoch_end(self, epoch, logs=None):
		# Stop Train when loss < MinLoss
		if logs['val_loss'] < MinLoss:
			print('Stop Training with Loss Checking on Epoch %d | Loss : %.5f' % ((epoch+1), logs['loss']))
			self.model.stop_training = True
		
		# Change per 5 Epoch
		if np.mod(epoch+1, ChangePer) == 0:
			Lr = K.get_value(self.model.optimizer.lr)
			Lr = (np.exp(-((1+epoch)/self.params['epochs'])) * Lr)
			K.set_value(self.model.optimizer.lr, Lr)
			print('Epoch : %d | Learning Rate : %.5f | Loss : %.5f' % (epoch+1, Lr, logs['loss']))

if __name__ == '__main__':
	main()
