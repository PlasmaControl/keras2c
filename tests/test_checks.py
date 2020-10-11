"""test_checks.py
This file is part of the test suite for keras2c
Implements tests for the checks run on the model before conversion
"""

#!/usr/bin/env python3

import unittest
import tensorflow.keras as keras
from keras2c import keras2c_main
import subprocess
import time
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestChecks(unittest.TestCase):
    """tests for model validity checking"""

    def test_is_model(self):
        model = np.arange(10)
        name = 'foo'
        self.assertRaises(ValueError, keras2c_main.k2c, model, name)

    def test_is_valid_cname(self):
        inshp = (10, 8)
        name = 'foobar'
        a = keras.layers.Input(inshp, name='f/oo')
        b = keras.layers.Dense(10)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

        name = '2foobar'
        a = keras.layers.Input(inshp)
        b = keras.layers.Dense(10)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

    def test_supported_layers(self):
        inshp = (10, 8)
        name = 'foobar'
        a = keras.layers.Input(inshp)
        b = keras.layers.Lambda(lambda x: x ** 2)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

    def test_activation_supported(self):
        inshp = (10, 8)
        name = 'foobar'
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(10, activation='elu',
                              recurrent_activation='selu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)


class TestConfigSupported(unittest.TestCase):

    def test_rnn_config_supported(self):
        inshp = (20, 10, 8)
        name = 'foobar'
        a = keras.layers.Input(batch_shape=inshp)
        b = keras.layers.LSTM(10, return_state=True,
                              stateful=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

    def test_shared_axes(self):
        inshp = (10, 8, 12)
        name = 'foobar'
        a = keras.layers.Input(inshp)
        b = keras.layers.PReLU(shared_axes=[1, 2])(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

    def test_data_format(self):
        inshp = (8, 12)
        name = 'foobar'
        a = keras.layers.Input(inshp)
        b = keras.layers.Conv1D(filters=10, kernel_size=2,
                                data_format='channels_first')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

    def test_broadcast_merge(self):
        inshp1 = (12,)
        inshp2 = (10, 12)
        name = 'foobar'
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Add()([a, b])
        model = keras.models.Model(inputs=[a, b], outputs=c)
        self.assertRaises(AssertionError, keras2c_main.k2c, model, name)

# keras itself doesnt support multiple axes for batchnorm, but tensorflow.keras does

    # def test_batch_norm_axis(self):
    #     inshp = (8, 12, 16)
    #     name = 'foobar'
    #     axis = (2, 3)
    #     a = keras.layers.Input(inshp)
    #     b = keras.layers.BatchNormalization(axis=axis)(a)
    #     model = keras.models.Model(inputs=a, outputs=b)
    #     self.assertRaises(AssertionError, keras2c_main.k2c, model, name)
