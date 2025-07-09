"""test_recurrent_layers.py
This file is part of the test suite for keras2c
Implements tests for recurrent layers
"""

#!/usr/bin/env python3

import unittest
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU
from tensorflow.keras.models import Model
from keras2c import keras2c_main
import time
from test_core_layers import build_and_run

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestRecurrentLayers(unittest.TestCase):
    """Tests for recurrent layers"""

    def test_SimpleRNN1(self):
        inshp = (4, 4, 46)
        units = 17
        a = Input(shape=inshp[1:], batch_size=inshp[0])
        b = SimpleRNN(units, activation='relu',
                      return_sequences=False,
                      stateful=True)(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_SimpleRNN2(self):
        inshp = (34, 17)
        units = 40
        a = Input(shape=inshp)
        b = SimpleRNN(units, go_backwards=True,
                      return_sequences=True,
                      activation='tanh')(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_SimpleRNN3(self):
        inshp = (34, 17)
        units = 40
        a = Input(shape=inshp)
        b = SimpleRNN(units, go_backwards=False,
                      return_sequences=True,
                      activation='tanh',
                      use_bias=False)(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LSTM1(self):
        inshp = (23, 32)
        units = 19
        a = Input(shape=inshp)
        b = LSTM(units, activation='relu',
                 return_sequences=False,
                 recurrent_activation='sigmoid')(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___LSTM1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LSTM2(self):
        inshp = (4, 80)
        units = 23
        a = Input(shape=inshp)
        b = LSTM(units, go_backwards=True,
                 return_sequences=True,
                 activation='sigmoid',
                 recurrent_activation='tanh')(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___LSTM2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LSTM3(self):
        inshp = (3, 4, 80)
        units = 23
        a = Input(shape=inshp[1:], batch_size=inshp[0])
        b = LSTM(units, go_backwards=False,
                 return_sequences=True,
                 activation='sigmoid',
                 recurrent_activation='tanh',
                 use_bias=False,
                 stateful=True)(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___LSTM3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GRU1(self):
        inshp = (12, 46)
        units = 17
        a = Input(shape=inshp)
        b = GRU(units, activation='softmax',
                recurrent_activation='softsign',
                return_sequences=False)(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___GRU1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GRU2(self):
        inshp = (5, 12, 46)
        units = 17
        a = Input(shape=inshp[1:], batch_size=inshp[0])
        b = GRU(units, activation='softplus',
                recurrent_activation='sigmoid',
                return_sequences=True,
                go_backwards=True,
                reset_after=True,
                stateful=True)(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___GRU2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GRU3(self):
        inshp = (12, 46)
        units = 17
        a = Input(shape=inshp)
        b = GRU(units, activation='softplus',
                recurrent_activation='sigmoid',
                return_sequences=True,
                go_backwards=False,
                reset_after=True,
                use_bias=False)(a)
        model = Model(inputs=a, outputs=b)
        name = 'test___GRU3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
