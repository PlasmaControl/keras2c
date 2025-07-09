"""test_layer_wrappers.py
This file is part of the test suite for keras2c
Implements tests for layer wrappers
"""

#!/usr/bin/env python3

import unittest
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, TimeDistributed, Conv1D, Conv2D
)
from keras2c import keras2c_main
from test_core_layers import build_and_run

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestWrappers(unittest.TestCase):
    """Tests for layer wrappers"""

    @unittest.skip  # no reason needed
    def test_Bidirectional1(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences=True),
                                input_shape=(5, 10), merge_mode='concat'))
        model.add(Bidirectional(LSTM(10, return_sequences=True),
                                merge_mode='mul'))
        model.add(Dense(5))
        name = 'test___Bidirectional1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skip  # no reason needed
    def test_Bidirectional2(self):
        model = Sequential()
        model.add(Dense(5, input_shape=(5, 10)))
        model.add(Bidirectional(LSTM(10, return_sequences=True),
                                merge_mode='ave'))
        model.add(Bidirectional(LSTM(10, return_sequences=True),
                                merge_mode='concat'))
        model.add(Dense(5))
        name = 'test___Bidirectional2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_TimeDistributed1(self):
        inputs = Input(shape=(8, 5))
        outputs = TimeDistributed(
            Dense(7, use_bias=True))(inputs)
        model = Model(inputs, outputs)
        name = 'test___TimeDistributed1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skip  # no reason needed
    def test_TimeDistributed2(self):
        model = Sequential()
        model.add(TimeDistributed(
            Dense(7, use_bias=True), input_shape=(8, 5)))
        model.add(Dense(10))
        name = 'test___TimeDistributed2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skip  # no reason needed
    def test_TimeDistributed3(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(8, 5)))
        model.add(TimeDistributed(
            Dense(7, use_bias=True)))
        name = 'test___TimeDistributed3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skip  # no reason needed
    def test_TimeDistributed4(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(8, 5)))
        model.add(TimeDistributed(
            Dense(7, use_bias=True)))
        model.add(Dense(10))
        name = 'test___TimeDistributed4' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skip  # no reason needed
    def test_TimeDistributed5(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(4, 8, 5)))
        model.add(TimeDistributed(
            Conv1D(7, kernel_size=2)))
        model.add(Dense(10))
        name = 'test___TimeDistributed5' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skip  # no reason needed
    def test_TimeDistributed6(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(4, 7, 8, 5)))
        model.add(TimeDistributed(
            Conv2D(7, kernel_size=2)))
        model.add(Dense(10))
        name = 'test___TimeDistributed6' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
