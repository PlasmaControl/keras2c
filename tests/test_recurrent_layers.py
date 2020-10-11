"""test_recurrent_layers.py
This file is part of the test suite for keras2c
Implements tests for recurrent layers
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


class TestRecurrentLayers(unittest.TestCase):
    """tests for recurrent layers"""

    def test_SimpleRNN1(self):
        inshp = (4, 4, 46)
        units = 17
        a = keras.layers.Input(batch_shape=inshp)
        b = keras.layers.SimpleRNN(units, activation='relu',
                                   return_sequences=False,
                                   stateful=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_SimpleRNN2(self):
        inshp = (34, 17)
        units = 40
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units, go_backwards=True,
                                   return_sequences=True,
                                   activation='tanh')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_SimpleRNN3(self):
        inshp = (34, 17)
        units = 40
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units, go_backwards=False,
                                   return_sequences=True,
                                   activation='tanh',
                                   use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LSTM1(self):
        inshp = (23, 32)
        units = 19
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(units, activation='relu',
                              return_sequences=False,
                              recurrent_activation='hard_sigmoid')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LSTM1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LSTM2(self):
        inshp = (4, 80)
        units = 23
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(units, go_backwards=True,
                              return_sequences=True,
                              activation='sigmoid',
                              recurrent_activation='tanh')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LSTM2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LSTM3(self):
        inshp = (3, 4, 80)
        units = 23
        a = keras.layers.Input(batch_shape=inshp)
        b = keras.layers.LSTM(units, go_backwards=False,
                              return_sequences=True,
                              activation='sigmoid',
                              recurrent_activation='tanh',
                              use_bias=False,
                              stateful=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LSTM3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GRU1(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units, activation='softmax',
                             recurrent_activation='softsign',
                             return_sequences=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GRU1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GRU2(self):
        inshp = (5, 12, 46)
        units = 17
        a = keras.layers.Input(batch_shape=inshp)
        b = keras.layers.GRU(units, activation='softplus',
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             go_backwards=True,
                             reset_after=True,
                             stateful=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GRU2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GRU3(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units, activation='softplus',
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             go_backwards=False,
                             reset_after=True,
                             use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GRU3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


if __name__ == "__main__":
    unittest.main()
