#!/usr/bin/env python3

"""
test_advanced_activation_layers.py
This file is part of the test suite for keras2c
Implements tests for advanced activation layers
"""

import unittest
from tensorflow import keras
from keras2c import keras2c_main
import time
from test_core_layers import build_and_run

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestAdvancedActivation(unittest.TestCase):
    """tests for advanced activation layers"""

    def test_LeakyReLU(self):
        inshp = (9, 7, 6, 3)
        alpha = 0.5
        a = keras.Input(shape=inshp)
        b = keras.layers.LeakyReLU(negative_slope=alpha)(a)
        model = keras.Model(inputs=a, outputs=b)
        name = 'test___LeakyReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_PReLU(self):
        inshp = (3, 6, 9, 3)
        a = keras.Input(shape=inshp)
        b = keras.layers.PReLU(alpha_initializer='glorot_uniform')(a)
        model = keras.Model(inputs=a, outputs=b)
        name = 'test___PReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ELU(self):
        inshp = (13, 6, 9, 13)
        alpha = 1.3
        a = keras.Input(shape=inshp)
        b = keras.layers.ELU(alpha=alpha)(a)
        model = keras.Model(inputs=a, outputs=b)
        name = 'test___ELU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ReLU(self):
        inshp = (12, 7, 9, 21)
        max_value = 1.0
        negative_slope = 1.0
        threshold = 0.3
        a = keras.Input(shape=inshp)
        b = keras.layers.ReLU(max_value=max_value,
                              negative_slope=negative_slope,
                              threshold=threshold)(a)
        model = keras.Model(inputs=a, outputs=b)
        name = 'test___ReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ThresholdedReLU(self):
        """Test conversion of ThresholdedReLU."""
        inshp = (4, 5, 6, 3)
        theta = 0.5
        a = keras.Input(shape=inshp)
        b = keras.layers.ThresholdedReLU(theta=theta)(a)
        model = keras.Model(inputs=a, outputs=b)
        name = 'test___ThresholdedReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
