## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os
import sys

def test(data, file_name, params, batch_size=128, test_temp=1):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    # print(data.test_data.shape)
    print(file_name)

    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.test_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))

    model.load_weights(file_name)


    predicted = model.predict(data.test_data)
    y_pred = tf.nn.softmax(predicted/test_temp)
    # print(y_pred)
    y = data.test_labels
    # print(y)

    m = tf.keras.metrics.SparseCategoricalAccuracy()
    m.update_state(np.argmax(y,axis=1), y_pred)
    print(m.result().numpy())

    return
T = sys.argv[1]
test(MNIST(), "models_ta/mnist-distilled-"+T+"_teacher", [32, 32, 64, 64, 200, 200])
test(MNIST(), "models_ta/mnist-distilled-"+T+"_assistant", [32, 32, 64, 64, 200, 200])
test(MNIST(), "models_ta/mnist-distilled-"+T, [32, 32, 64, 64, 200, 200])

test(CIFAR(), "models_ta/cifar-distilled-"+T+"_teacher", [64, 64, 128, 128, 256, 256])
test(CIFAR(), "models_ta/cifar-distilled-"+T+"_assistant", [64, 64, 128, 128, 256, 256])
test(CIFAR(), "models_ta/cifar-distilled-"+T, [64, 64, 128, 128, 256, 256])
