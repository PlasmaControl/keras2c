"""test_checks.py
This file is part of the test suite for keras2c
Implements tests for the checks run on the model before conversion
"""

#!/usr/bin/env python3

import unittest
from tensorflow import keras
from keras2c import keras2c_main
import numpy as np

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestChecks(unittest.TestCase):
    """tests for model validity checking"""

    def test_is_model(self):
        """Non-Model input should raise ``ValueError``."""
        model = np.arange(10)
        name = 'foo'
        with self.assertRaises(ValueError):
            keras2c_main.k2c(model, name)

    def test_is_valid_cname(self):
        """Invalid C identifier should trigger ``AssertionError``."""
        inshp = (10, 8)
        name = '2foobar'
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Dense(10)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        with self.assertRaises(AssertionError):
            keras2c_main.k2c(model, name)

    def test_supported_layers(self):
        """Unsupported layers should fail the conversion checks."""
        inshp = (10, 8)
        name = 'foobar'
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Lambda(lambda x: x ** 2)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        with self.assertRaises(AssertionError):
            keras2c_main.k2c(model, name)

    def test_activation_supported(self):
        """Previously unsupported activations should now convert without errors."""
        inshp = (10, 8)
        name = 'foobar_1'
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.LSTM(
            10, activation='gelu', recurrent_activation='swish'
        )(a)
        model = keras.models.Model(inputs=a, outputs=b)
        keras2c_main.k2c(model, name)


class TestConfigSupported(unittest.TestCase):

    def test_rnn_config_supported(self):
        inshp = (20, 80)
        batch_size = 32
        name = 'foobar'
        a = keras.layers.Input(shape=inshp, batch_size=batch_size)
        b = keras.layers.LSTM(10, return_state=True,
                              stateful=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        with self.assertRaises(AssertionError):
            keras2c_main.k2c(model, name)

    def test_shared_axes(self):
        """PReLU with shared axes is currently unsupported."""
        inshp = (10, 8, 12)
        name = 'foobar'
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.PReLU(shared_axes=[1, 2])(a)
        model = keras.models.Model(inputs=a, outputs=b)
        with self.assertRaises(AssertionError):
            keras2c_main.k2c(model, name)

    def test_data_format(self):
        """Unsupported data format should trigger ``AssertionError``."""
        inshp = (8, 12)
        name = 'foobar'
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Conv1D(
            filters=10, kernel_size=2, data_format='channels_first'
        )(a)
        model = keras.models.Model(inputs=a, outputs=b)
        with self.assertRaises(AssertionError):
            keras2c_main.k2c(model, name)

    def test_broadcast_merge(self):
        """Merging tensors of different sizes should fail."""
        inshp1 = (12,)
        inshp2 = (10, 12)
        name = 'foobar'
        a = keras.layers.Input(shape=inshp1)
        b = keras.layers.Input(shape=inshp2)
        c = keras.layers.Add()([a, b])
        model = keras.models.Model(inputs=[a, b], outputs=c)
        with self.assertRaises(AssertionError):
            keras2c_main.k2c(model, name)

    # Keras 3 does not support multiple axes for BatchNormalization 'axis' parameter
    # Uncomment the test below if you need to check for unsupported axes in BatchNormalization

    # def test_batch_norm_axis(self):
    #     inshp = (8, 12, 16)
    #     name = 'foobar'
    #     axis = (2, 3)
    #     a = keras.layers.Input(shape=inshp)
    #     b = keras.layers.BatchNormalization(axis=axis)(a)
    #     model = keras.models.Model(inputs=a, outputs=b)
    #     with self.assertRaises(AssertionError):
    #         keras2c_main.k2c(model, name)
