"""test_pooling_layers.py
This file is part of the test suite for keras2c
Implements tests for pooling layers
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


class TestPoolingLayers(unittest.TestCase):
    """tests for pooling layers"""

    def test_MaxPooling1D1(self):
        inshp = (23, 29)
        pool_size = 3
        strides = 2
        padding = 'valid'
        a = keras.layers.Input(inshp)
        b = keras.layers.MaxPooling1D(pool_size=pool_size,
                                      strides=strides,
                                      padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___MaxPooling1D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_MaxPooling1D2(self):
        inshp = (13, 19)
        pool_size = 2
        strides = 2
        padding = 'same'
        a = keras.layers.Input(inshp)
        b = keras.layers.MaxPooling1D(pool_size=pool_size,
                                      strides=strides,
                                      padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___MaxPooling1D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_AveragePooling1D1(self):
        inshp = (23, 29)
        pool_size = 2
        strides = 3
        padding = 'valid'
        a = keras.layers.Input(inshp)
        b = keras.layers.AveragePooling1D(pool_size=pool_size,
                                          strides=strides,
                                          padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___AveragePooling1D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_AveragePooling1D2(self):
        inshp = (13, 19)
        pool_size = 3
        strides = 1
        padding = 'same'
        a = keras.layers.Input(inshp)
        b = keras.layers.AveragePooling1D(pool_size=pool_size,
                                          strides=strides,
                                          padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___AveragePooling1D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_MaxPooling2D1(self):
        inshp = (23, 29, 7)
        pool_size = (3, 2)
        strides = (2, 1)
        padding = 'valid'
        a = keras.layers.Input(inshp)
        b = keras.layers.MaxPooling2D(pool_size=pool_size,
                                      strides=strides,
                                      padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___MaxPooling2D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_MaxPooling2D2(self):
        inshp = (13, 19, 4)
        pool_size = (2, 4)
        strides = (2, 3)
        padding = 'same'
        a = keras.layers.Input(inshp)
        b = keras.layers.MaxPooling2D(pool_size=pool_size,
                                      strides=strides,
                                      padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___MaxPooling2D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_AveragePooling2D1(self):
        inshp = (18, 25, 7)
        pool_size = (2, 3)
        strides = (2, 2)
        padding = 'valid'
        a = keras.layers.Input(inshp)
        b = keras.layers.AveragePooling2D(pool_size=pool_size,
                                          strides=strides,
                                          padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___AveragePooling2D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_AveragePooling2D2(self):
        inshp = (23, 29, 5)
        pool_size = (2, 4)
        strides = (3, 1)
        padding = 'same'
        a = keras.layers.Input(inshp)
        b = keras.layers.AveragePooling2D(pool_size=pool_size,
                                          strides=strides,
                                          padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___AveragePooling2D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GlobalAveragePooling1D(self):
        inshp = (16, 11)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalAveragePooling1D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalAveragePooling1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GlobalMaxPooling1D(self):
        inshp = (31, 21)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalMaxPooling1D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalMaxPooling1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GlobalAveragePooling2D(self):
        inshp = (16, 11, 13)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalAveragePooling2D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalAveragePooling2D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GlobalMaxPooling2D(self):
        inshp = (31, 21, 5)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalMaxPooling2D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalMaxPooling2D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GlobalAveragePooling3D(self):
        inshp = (16, 11, 4, 5)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalAveragePooling3D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalAveragePooling3D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_GlobalMaxPooling3D(self):
        inshp = (31, 21, 6, 7)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalMaxPooling3D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalMaxPooling3D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
