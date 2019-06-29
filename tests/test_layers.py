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

ccflags = '-g -O0 -std=c99 -fprofile-arcs -ftest-coverage -I./include/'


class TestRecurrentLayers(unittest.TestCase):
    """tests for recurrent layers"""

    def test_SimpleRNN(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test_SimpleRNN' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        cc = 'gcc ' + ccflags + ' -o ' + name + ' ' + name + '_test_suite.c -lm'
        subprocess.run(cc.split())
        rcode = subprocess.run(['./' + name])
        self.assertEqual(rcode.returncode, 0)
        subprocess.run(['rm', './' + name, './' + name + '.h',
                        './' + name + '_test_suite.c'])

    def test_LSTM(self):
        inshp = (23, 32)
        units = 19
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(units)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test_LSTM' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        cc = 'gcc ' + ccflags + ' -o ' + name + ' ' + name + '_test_suite.c -lm'
        subprocess.run(cc.split())
        rcode = subprocess.run(['./' + name])
        self.assertEqual(rcode.returncode, 0)
        subprocess.run(['rm', './' + name, './' + name + '.h',
                        './' + name + '_test_suite.c'])

    def test_GRU(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test_GRU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        cc = 'gcc ' + ccflags + ' -o ' + name + ' ' + name + '_test_suite.c -lm'
        subprocess.run(cc.split())
        rcode = subprocess.run(['./' + name])
        self.assertEqual(rcode.returncode, 0)
        subprocess.run(['rm', './' + name, './' + name + '.h',
                        './' + name + '_test_suite.c'])


if __name__ == "__main__":
    unittest.main()
