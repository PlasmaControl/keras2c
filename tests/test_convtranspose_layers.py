"""test_convolution_layers.py
This file is part of the test suite for keras2c
Implements tests for convolution layers
"""

#!/usr/bin/env python3

import time
import unittest

import keras
import numpy as np
from test_core_layers import build_and_run

from keras2c import keras2c_main

__author__ = "Anchal Gupta"
__copyright__ = "Copyright 2024, Anchal Gupta"
__license__ = "MIT"
__maintainer__ = "Anchal Gupta, https://github.com/anchal-physics/keras2c"
__email__ = "guptaa@fusion.gat.com"


class TestConvolutionTransposeLayers(unittest.TestCase):
    """tests for convolution transpose layers"""

    def test_Conv1DTranspose1(self):
        for tno in range(10):
            # Variable names match k2c_conv1d_transpose in k2c_conv_transpose_layer.c
            n_height = np.random.randint(2, 50)
            n_channels = np.random.randint(1, 50)
            n_filters = np.random.randint(1, 50)
            k_size = np.random.randint(1, n_height)
            stride = np.random.randint(1, max(k_size, 2))
            inshp = (n_height, n_channels)
            if tno % 2 == 0:
                padding = "valid"
            else:
                padding = "same"
            dilation_rate = 1
            activation = None
            a = keras.layers.Input(inshp)
            b = keras.layers.Conv1DTranspose(
                filters=n_filters,
                kernel_size=k_size,
                strides=stride,
                padding=padding,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=False,
            )(a)
            model = keras.models.Model(inputs=a, outputs=b)
            name = "test___Conv1DTranspose1" + str(int(time.time()))
            keras2c_main.k2c(model, name)
            rcode = build_and_run(name)
            self.assertEqual(rcode, 0)

    def test_Conv2DTranspose1(self):
        for tno in range(10):
            # Variable names match k2c_conv2d_transpose in k2c_conv_transpose_layer.c
            in_rows = np.random.randint(2, 25)
            in_cols = np.random.randint(2, 25)
            in_channels = np.random.randint(1, 25)
            n_filters = np.random.randint(1, 25)
            k_rows = np.random.randint(1, in_rows)
            k_cols = np.random.randint(1, in_cols)
            stride_h = np.random.randint(1, max(k_rows, 2))
            stride_w = np.random.randint(1, max(k_cols, 2))
            inshp = (in_rows, in_cols, in_channels)
            if tno % 2 == 0:
                padding = "valid"
            else:
                padding = "same"
            dilation_rate = 1
            activation = None
            a = keras.layers.Input(inshp)
            b = keras.layers.Conv2DTranspose(
                filters=n_filters,
                kernel_size=(k_rows, k_cols),
                strides=(stride_h, stride_w),
                padding=padding,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=False,
            )(a)
            model = keras.models.Model(inputs=a, outputs=b)
            name = "test___Conv2DTranspose1" + str(int(time.time()))
            keras2c_main.k2c(model, name)
            rcode = build_and_run(name)
            self.assertEqual(rcode, 0)
