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
from datetime import datetime

__author__ = 'aGn'


def tf_keras_model(hidden, input_shape, num_class, lr):
    """
    Make tf.keras model by a hidden layer
    :param hidden: Number of Hidden Neuron
    :param input_shape:
    :param num_class:
    :param lr:
    :return: A tf_keras_model.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(hidden, input_shape=[input_shape, ],
                                    activation='tanh',
                                    use_bias=True))
    model.add(tf.keras.layers.Dense(num_class,
                                    activation='sigmoid',
                                    use_bias=True))
    # Adam Optimizer with Binary Cross Entropy Loss Function
    model.compile(tf.keras.optimizers.Adam(lr=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_classification(model, moon_x, moon_y):
    """For Contour Prediction"""

    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 200),
                       np.linspace(-1.5, 1.5, 200))
    xy = np.c_[x.ravel(), y.ravel()]
    predict = model.predict(xy)
    predict = predict.reshape(x.shape)
    plt.cla()
    plt.contourf(x, y, predict, cmap='bwr', alpha=0.2)
    plt.plot(moon_x[moon_y == 0, 0], moon_x[moon_y == 0, 1], 'ob')
    plt.plot(moon_x[moon_y == 1, 0], moon_x[moon_y == 1, 1], 'or')
    plt.show()


def main():
    # Make Moon Data
    moon_x, moon_y = \
        datasets.make_circles(
            n_samples=500,
            factor=.2,
            noise=.1,
            random_state=0)

    # Make tf_keras Model
    model = tf_keras_model(hidden=20, input_shape=2, num_class=1, lr=.5)

    '''TensorBoardCb'''
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train Model
    model.fit(moon_x, moon_y,  # Inputs
              epochs=20,
              batch_size=128,
              verbose=1,  # Show result method
              validation_split=.2,  # Set 20% Train Data for Validation Check
              callbacks=[tensor_board_callback])

    # Plot Result
    plot_classification(model, moon_x, moon_y)


if __name__ == '__main__':
    main()
