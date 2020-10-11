"""test_layer_wrappers.py
This file is part of the test suite for keras2c
Implements tests for layer wrappers
"""

#!/usr/bin/env python3

from test_core_layers import build_and_run
import numpy as np
import os
import time
import subprocess
from keras2c import keras2c_main
import unittest
import tensorflow.keras as keras
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestWrappers(unittest.TestCase):
    """tests for layer wrappers"""

    def test_Bidirectional1(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences=True),
                                             input_shape=(5, 10), merge_mode='concat'))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences=True),
                                             merge_mode='mul'))
        model.add(keras.layers.Dense(5))
        model.build()
        name = 'test___Bidirectional1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Bidirectional2(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(5, input_shape=(5, 10)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences=True),
                                             merge_mode='ave'))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences=True),
                                             merge_mode='concat'))
        model.add(keras.layers.Dense(5))
        model.build()
        name = 'test___Bidirectional2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed1(self):

        inputs = keras.layers.Input(shape=(8, 5))
        outputs = keras.layers.TimeDistributed(
            keras.layers.Dense(7, use_bias=True,))(inputs)
        model = keras.models.Model(inputs, outputs)
        model.build((None, 8, 5))
        name = 'test___TimeDistributed1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed2(self):
        model = keras.models.Sequential()
        model.add(keras.layers.TimeDistributed(
            keras.layers.Dense(7, use_bias=True), input_shape=(8, 5)))
        model.add(keras.layers.Dense(10))
        model.build()
        name = 'test___TimeDistributed2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed3(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_shape=(8, 5)))
        model.add(keras.layers.TimeDistributed(
            keras.layers.Dense(7, use_bias=True)))
        model.build()
        name = 'test___TimeDistributed3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed4(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_shape=(8, 5)))
        model.add(keras.layers.TimeDistributed(
            keras.layers.Dense(7, use_bias=True)))
        model.add(keras.layers.Dense(10))
        model.build()
        name = 'test___TimeDistributed4' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed5(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_shape=(4, 8, 5)))
        model.add(keras.layers.TimeDistributed(
            keras.layers.Conv1D(7, kernel_size=2)))
        model.add(keras.layers.Dense(10))
        model.build()
        name = 'test___TimeDistributed5' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed6(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_shape=(4, 7, 8, 5)))
        model.add(keras.layers.TimeDistributed(
            keras.layers.Conv2D(7, kernel_size=2)))
        model.add(keras.layers.Dense(10))
        model.build()
        name = 'test___TimeDistributed6' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
