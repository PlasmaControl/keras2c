"""test_layers.py
This file is part of the test suite for keras2c
Author: Rory Conlin
License: GNU GPLv3
"""

#!/usr/bin/env python3

import unittest
import keras
from keras2c import keras2c_main
import subprocess
import time


class TestRecurrentLayers(unittest.TestCase):
    """tests for recurrent layers"""

    def test_SimpleRNN(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'SimpleRNN' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        subprocess.run(['gcc', '-fprofile-arcs', '-ftest-coverage',
                        '-I./include/', '-o', name, name + '_test_suite.c', '-lm'])
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
        name = 'LSTM' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        subprocess.run(['gcc', '-fprofile-arcs', '-ftest-coverage',
                        '-I./include/', '-o', name, name + '_test_suite.c', '-lm'])
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
        name = 'GRU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        subprocess.run(['gcc', '-fprofile-arcs', '-ftest-coverage',
                        '-I./include/', '-o', name, name + '_test_suite.c', '-lm'])
        rcode = subprocess.run(['./' + name])
        self.assertEqual(rcode.returncode, 0)
        subprocess.run(['rm', './' + name, './' + name + '.h',
                        './' + name + '_test_suite.c'])


if __name__ == "__main__":
    unittest.main()
