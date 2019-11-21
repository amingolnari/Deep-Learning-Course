"""
github : https://github.com/agn-7/Deep-Learning-Course
Author : Benyamin Jafari (aGn)
TensorFlow Version : 2.0.0b1
Date : Nov-2019

Using Callback in tf.keras
"""

"""
Stop training with validation loss check
Change learning rate per `n` epoch
Save learning rate per train epoch
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import datasets
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from datetime import datetime

__author__ = 'aGn'


def KerasModel(Hidden, InputShape, NumClass, lr):
    """
    Make tf.keras model by a hidden layer
    :param Hidden: Number of Hidden Neuron
    :param InputShape:
    :param NumClass:
    :param lr:
    :return:
    """
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
    model = KerasModel(Hidden, InputShape, NumClass, lr)

    '''TensorBoardCb'''
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train Model
    model.fit(moon_X, moon_Y, # Inputs
              epochs=Epochs,
              batch_size=Batch,
              verbose=0,  # Show result method
              validation_split=.2,  # Set 20% Train Data for Validation Check
              callbacks=[tensor_board_callback])

    # Plot Result
    PlotClassification(model, moon_X, moon_Y)
  
    # Plot Learning Rate in Train Epochs
    plt.plot(call.learninrate, 'ro')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.xlim(-0.1, len(call.learninrate)+.1)
    plt.show()


if __name__ == '__main__':
    main()
