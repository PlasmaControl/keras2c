"""test_malloc.py
This file is part of the test suite for keras2c
Implements tests for dynamic memory allocation
"""

#!/usr/bin/env python3

import unittest
import tensorflow.keras as keras
from keras2c import keras2c_main
import subprocess
import time
import os
from test_core_layers import build_and_run
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestMalloc(unittest.TestCase):
    """tests for dynamic memory allocation for large weight tensors"""

    def test_Malloc1(self):
        inshp = (21, 4, 9)
        units = 45
        a = keras.layers.Input(inshp)
        b = keras.layers.Dense(units, activation='relu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Malloc1' + str(int(time.time()))
        keras2c_main.k2c(model, name, malloc=True)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Malloc2(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(8, (3, 3), padding='same',
                                      input_shape=(32, 32, 3)))
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Conv2D(8, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(8, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(8, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(20))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Activation('softmax'))
        name = 'test___Malloc2' + str(int(time.time()))
        keras2c_main.k2c(model, name, malloc=True)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


if __name__ == "__main__":
    unittest.main()
