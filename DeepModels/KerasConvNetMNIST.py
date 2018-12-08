"""
github : https://github.com/amingolnari/Deep-Learning-Course
Author : Amin Golnari
Keras Version : 2.2.4
Date : 4/12/2018

Keras CNN Classification on MNIST Data
Code 301
"""

## If your GPU is AMD , you can use PlaidML Backend
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST Data (Download for First)
def LoadData():
    (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
    Xtrain = Xtrain.reshape(60000, 28, 28, 1).astype('float32')
    Xtrain = Xtrain / 255 # Normalize to 0-1
    Xtest = Xtest.reshape(10000, 28, 28, 1).astype('float32')
    Xtest = Xtest / 255
    Ytrain = to_categorical(Ytrain, 10) # for exam Label 2 : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    Ytest = to_categorical(Ytest, 10)

    return (Xtrain, Xtest), (Ytrain, Ytest)

def BuildModel():
    model = Sequential()
    model.add(Conv2D(filters = 128, kernel_size = (5, 5),
                     activation = 'relu',
                     padding = 'same',
                     input_shape = (28, 28, 1))) # Just First (Input) Layer Need Init input_shape
    model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                     padding = 'same',
                     activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                     activation = 'relu',
                     padding = 'same'))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Dense(300, activation = 'tanh')) # Hidden Layer #1
    model.add(Dense(200, activation = 'sigmoid')) # Hidden Layer #1
    model.add(Dense(10, activation = 'softmax'))# Output Layer

    return model

def PlotHistory(history):
  plt.title('Keras Model loss/accuracy')
  plt.ylabel('loss/accuracy')
  plt.xlabel('epochs')
  #  Accuracy
  plt.plot(history.history['acc'], '.-')
  plt.plot(history.history['val_acc'], '-.')
  # Loss
  plt.plot(history.history['loss'], '-*')
  plt.plot(history.history['val_loss'], '*-')
  plt.legend(['Train loss', 'Validation loss', 'Train acc', 'Validation acc'], loc='upper right')
  plt.grid(True, linestyle = '-.')
  plt.tick_params(labelcolor = 'b', labelsize = 'medium', width = 3)
  fig = plt.gcf()
  fig.savefig('images/model loss.jpg')
  plt.show()
  return

def main():
    (Xtrain, Xtest), (Ytrain, Ytest) = LoadData()

    model = BuildModel()
    model.summary()

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = SGD(lr = 0.1),
                  metrics = ['accuracy'])
    History = model.fit(Xtrain, Ytrain,
                        batch_size = 256,
                        epochs = 100,
                        validation_split = .3)
    PlotHistory(History)
    
    score, acc = model.evaluate(Xtest, Ytest)
    print('Test Accuracy : ', acc)

if __name__ == "__main__":
    main()
