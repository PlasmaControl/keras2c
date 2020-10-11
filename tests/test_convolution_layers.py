"""test_convolution_layers.py
This file is part of the test suite for keras2c
Implements tests for convolution layers
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


class TestConvolutionLayers(unittest.TestCase):
    """tests for convolution layers"""

    def test_Conv3D1(self):
        inshp = (25, 32, 3, 4)
        filters = 13
        kernel_size = (3, 4, 2)
        strides = (2, 3, 2)
        padding = 'valid'
        dilation_rate = 1
        activation = 'relu'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv3D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv3D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Conv3D2(self):
        inshp = (25, 32, 3, 4)
        filters = 13
        kernel_size = (3, 4, 3)
        strides = 1
        padding = 'same'
        dilation_rate = (1, 2, 3)
        activation = 'relu'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv3D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv3D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Conv2D1(self):
        inshp = (25, 32, 3)
        filters = 13
        kernel_size = (3, 4)
        strides = (2, 3)
        padding = 'valid'
        dilation_rate = 1
        activation = 'relu'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv2D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Conv2D2(self):
        inshp = (13, 23, 10)
        filters = 17
        kernel_size = 2
        strides = 1
        padding = 'same'
        dilation_rate = (3, 2)
        activation = 'sigmoid'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=True,
                                bias_initializer='glorot_uniform')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv2D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Conv1D1(self):
        inshp = (25, 32)
        filters = 13
        kernel_size = 3
        strides = 2
        padding = 'valid'
        dilation_rate = 1
        activation = 'relu'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv1D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv1D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Conv1D2(self):
        inshp = (13, 23)
        filters = 17
        kernel_size = 4
        strides = 1
        padding = 'same'
        dilation_rate = 3
        activation = 'sigmoid'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv1D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=True,
                                bias_initializer='glorot_uniform')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv1D2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Conv1D3(self):
        inshp = (8, 32)
        filters = 17
        kernel_size = 4
        strides = 1
        padding = 'causal'
        dilation_rate = 1
        activation = 'tanh'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv1D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                use_bias=True,
                                bias_initializer='glorot_uniform')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Conv1D3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


class TestPadding(unittest.TestCase):

    def test_ZeroPad1D(self):
        inshp = (10, 12)
        pad_top = 3
        pad_bottom = 1
        a = keras.layers.Input(inshp)
        b = keras.layers.ZeroPadding1D(padding=(pad_top, pad_bottom))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ZeroPad1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ZeroPad2D(self):
        inshp = (10, 12, 5)
        pad_top = 3
        pad_bottom = 1
        pad_left = 4
        pad_right = 3
        a = keras.layers.Input(inshp)
        b = keras.layers.ZeroPadding2D(
            padding=((pad_top, pad_bottom), (pad_left, pad_right)))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ZeroPad2D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ZeroPad3D(self):
        inshp = (10, 12, 5, 4)
        pad_top = 3
        pad_bottom = 1
        pad_left = 4
        pad_right = 3
        pad_front = 2
        pad_back = 4
        a = keras.layers.Input(inshp)
        b = keras.layers.ZeroPadding3D(
            padding=((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ZeroPad3D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


class TestCropping(unittest.TestCase):

    def test_Cropping1D(self):
        inshp = (10, 12)
        crop_top = 3
        crop_bottom = 1
        a = keras.layers.Input(inshp)
        b = keras.layers.Cropping1D(cropping=(crop_top, crop_bottom))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Cropping1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Cropping2D(self):
        inshp = (10, 12, 5)
        crop_top = 3
        crop_bottom = 1
        crop_left = 4
        crop_right = 3
        a = keras.layers.Input(inshp)
        b = keras.layers.Cropping2D(
            cropping=((crop_top, crop_bottom), (crop_left, crop_right)))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Cropping2D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Cropping3D(self):
        inshp = (10, 12, 5, 4)
        crop_top = 3
        crop_bottom = 1
        crop_left = 4
        crop_right = 3
        crop_front = 2
        crop_back = 0
        a = keras.layers.Input(inshp)
        b = keras.layers.Cropping3D(
            cropping=((crop_top, crop_bottom), (crop_left, crop_right), (crop_front, crop_back)))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Cropping3D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


class TestUpSampling(unittest.TestCase):

    def test_UpSampling1D(self):
        inshp = (4, 10)
        a = keras.layers.Input(inshp)
        b = keras.layers.UpSampling1D(3)(a)
        model = keras.models.Model(a, b)
        name = 'test___UpSampling1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_UpSampling2D(self):
        inshp = (4, 5, 10)
        a = keras.layers.Input(inshp)
        b = keras.layers.UpSampling2D((3, 4))(a)
        model = keras.models.Model(a, b)
        name = 'test___UpSampling2D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_UpSampling3D(self):
        inshp = (4, 5, 10, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.UpSampling3D((3, 4, 2))(a)
        model = keras.models.Model(a, b)
        name = 'test___UpSampling3D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
