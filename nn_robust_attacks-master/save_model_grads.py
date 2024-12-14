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
import pandas as pd

def model_grad(data, file_name, params, batch_size=128, test_temp=1):
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
    avg_grad = []
    # print(data.test_data[0])
    for i in tqdm(range(len(data.test_data))):
      inp = tf.Variable(tf.expand_dims(data.test_data[i], axis=0))
      with tf.GradientTape() as tape:
        pred = model(inp)
      grads = tape.gradient(pred, inp)
      avg_grad.append(tf.reduce_mean(tf.math.abs(grads)).numpy())


    return avg_grad

grad_dict = {}
T = sys.argv[1]
test(MNIST(), "models_ta/mnist-distilled-"+T+"_teacher", [32, 32, 64, 64, 200, 200])
test(MNIST(), "models_ta/mnist-distilled-"+T+"_assistant", [32, 32, 64, 64, 200, 200])
test(MNIST(), "models_ta/mnist-distilled-"+T, [32, 32, 64, 64, 200, 200])

test(CIFAR(), "models_ta/cifar-distilled-"+T+"_teacher", [64, 64, 128, 128, 256, 256])
test(CIFAR(), "models_ta/cifar-distilled-"+T+"_assistant", [64, 64, 128, 128, 256, 256])
test(CIFAR(), "models_ta/cifar-distilled-"+T, [64, 64, 128, 128, 256, 256])
