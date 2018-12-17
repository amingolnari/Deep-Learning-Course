"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
Keras Version : 2.2.4
Date : 15/12/2018

Keras Generative Adversarial Network (GAN) on MNIST Data
Code 304
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist

def main():
	size = 28
	Batch = 256
	Epochs = 500
	NumIm = 36
	NumSample = 10000
	dLoss = []
	gLoss = []
	(X_train, _), (_, _) = mnist.load_data()
	X_train = X_train.reshape(60000, size*size*1).astype('float32')
	X_train = X_train / 255
	X_train = X_train[:NumSample]
	
	Idx = np.random.randint(0, X_train.shape[0], NumIm)
	ShowImages(X_train[Idx], (int)(np.sqrt(NumIm)), 0, 1)
	
	dis = discriminator(size = size)
	Trainable(dis, False)
	gen = generator(size = size)
	gan = Sequential()
	gan.add(gen)
	gan.add(dis)
	gan.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = .0001), metrics = None)
	gan.summary()

	NumIteration = int(X_train.shape[0] / Batch)
	for it in tqdm(range(Epochs)):
		gloss, dloss = 0, 0
		for batch in range(NumIteration):
			Real = X_train[batch*Batch:(batch+1)*Batch, :]
			noise = (2 * np.random.uniform(0, 1, size = [len(Real), 100]).astype('float32')) - 1
			Fake = gen.predict(noise)
			RealT = np.ones(len(Real), dtype = 'float32')
			FakeT = np.zeros(len(Fake), dtype = 'float32')
			
			Input = np.concatenate((Real, Fake))
			Target = np.concatenate((RealT, FakeT))
			Idx = np.random.randint(0, Input.shape[0], Input.shape[0])
			Input, Target = Input[Idx], Target[Idx]
			
			# dis.trainable = True
			Trainable(dis, True)
			dloss += dis.train_on_batch(Input, Target)
			Trainable(dis, False)
			# dis.trainable = False
			
			noise = (2 * np.random.uniform(0, 1, size = [len(Real), 100]).astype('float32')) - 1
			FakeT = np.ones(len(Real), dtype = 'float32')
			gloss += gan.train_on_batch(noise, FakeT)
			
			if (np.mod((it+1), 50) == 0 or it == 0) and batch == (NumIteration-1):
				PlotLoss(gloss = gLoss, dloss = dLoss)
				Idx = np.random.randint(0, Fake.shape[0], NumIm)
				ShowImages(Fake[Idx], (int)(np.sqrt(NumIm)), (it + 1), 0)
				gen.save('models/' + (it+1).__str__() + ' Generator.h5')
				dis.save('models/' + (it+1).__str__() + ' Discriminator.h5')
		dLoss.append(np.sum(dloss)/NumIteration)
		gLoss.append(np.sum(gloss)/NumIteration)
	return

def Trainable(net, Train = True):
	net.trainable = Train
	for Layer in net.layers:
		Layer.trainable = Train
	return

def generator(size = 28):
	model = Sequential()
	model.add(Dense(units = 256, input_shape = [100,]))
	model.add(Activation('relu'))
	model.add(Dense(units = 512))
	model.add(Activation('relu'))
	model.add(Dense(units = 1024))
	model.add(LeakyReLU(alpha = .2))
	model.add(Dense(units = size*size*1))
	model.add(Activation('tanh'))
	model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = .0001), metrics = None)
	model.summary()
	return model

def discriminator(size = 28):
	model = Sequential()
	model.add(Dense(units = 1024,
	                input_shape = [size*size*1],
	                activation = 'relu'))
	model.add(Dense(units = 512))
	model.add(LeakyReLU(alpha = .2))
	model.add(Dense(units = 256))
	model.add(LeakyReLU(alpha = .2))
	model.add(Dropout(rate = .2))
	model.add(Dense(units = 1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = .0001), metrics = None)
	model.summary()
	return model

def PlotLoss(dloss, gloss):
	plt.clf()
	plt.plot(gloss, label = 'Generator', color = 'red')
	plt.plot(dloss, label = 'Discriminator', color = 'blue')
	plt.title('Digits - Train GAN')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss Function')
	plt.savefig('images/Loss Function.png')
	plt.show()
	return

def ShowImages(Data, size, num, mode):
	k = 0
	Image = []
	for i in range(size):
		data = []
		for j in range(size):
			temp = Data[k]
			temp += 1
			temp *= 127.5
			data.append(temp.reshape(28, 28))
			k += 1
		Image.append(np.hstack(data))
	Image = np.vstack(Image)
	if mode == 1:
		name = 'Real'
	else:
		name = 'Fake'
	plt.imshow(Image.astype('uint8'), cmap = 'gray')
	plt.title(name + ' Data ' + num.__str__())
	plt.axis('off')
	plt.savefig('images/' + name + ' Data in Epoch ' + num.__str__() + '.png')
	plt.show()


if __name__ == "__main__":
	main()
