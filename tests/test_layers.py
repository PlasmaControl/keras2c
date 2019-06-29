"""test_layers.py
This file is part of the test suite for keras2c
Implements tests for individual layers
"""

#!/usr/bin/env python3

import unittest
import keras
from keras2c import keras2c_main
import subprocess
import time

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def build_and_run(name):
    ccflags = '-g -O0 -std=c99 -fprofile-arcs -ftest-coverage -I./include/'
    cc = 'gcc ' + ccflags + ' -o ' + name + ' ' + name + '_test_suite.c -lm'
    subprocess.run(cc.split())
    rcode = subprocess.run(['./' + name])
    # subprocess.run(['rm', './' + name, './' + name + '.h',
    #                 './' + name + '_test_suite.c'])
    return rcode


class TestRecurrentLayers(unittest.TestCase):
    """tests for recurrent layers"""

    def test_SimpleRNN1(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units, activation='relu',
                                   return_sequences=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)

    def test_LSTM1(self):
        inshp = (23, 32)
        units = 19
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(units, activation='relu',
                              return_sequences=True,
                              recurrent_activation='hard_sigmoid')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LSTM1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)

    def test_GRU1(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units, activation='softmax',
                             recurrent_activation='softsign',
                             return_sequences=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GRU1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_GRU2(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units, activation='softplus',
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             go_backwards=True,
                             reset_after=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GRU2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)


if __name__ == "__main__":
    unittest.main()
