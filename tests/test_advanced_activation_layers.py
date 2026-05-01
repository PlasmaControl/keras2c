"""test_advanced_activation_layers.py
This file is part of the test suite for keras2c
Implements tests for advanced activation layers
"""

#!/usr/bin/env python3

import unittest
import keras
from keras2c import keras2c_main
import time
from test_core_layers import build_and_run


def _has_activation(name):
    """True if the installed Keras recognizes the given activation identifier."""
    try:
        keras.activations.get(name)
        return True
    except (ValueError, TypeError):
        return False


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestAdvancedActivation(unittest.TestCase):
    """tests for advanced activation layers"""

    def test_swish(self):
        inshp = (9, 7, 6, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('swish')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___swish' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_LeakyReLU(self):
        inshp = (9, 7, 6, 3)
        alpha = 0.5
        a = keras.layers.Input(inshp)
        b = keras.layers.LeakyReLU(alpha=alpha)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LeakyReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_PReLU(self):
        inshp = (3, 6, 9, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.PReLU(alpha_initializer='glorot_uniform')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___PReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ELU(self):
        inshp = (13, 6, 9, 13)
        alpha = 1.3
        a = keras.layers.Input(inshp)
        b = keras.layers.ELU(alpha=alpha)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ELU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ThresholdedReLU(self):
        inshp = (3, 6, 19, 11)
        theta = 0.3
        a = keras.layers.Input(inshp)
        b = keras.layers.ReLU(threshold=theta)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ThresholdedReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ReLU(self):
        inshp = (12, 7, 9, 21)
        max_value = 1.0
        negative_slope = 1.0
        threshold = 0.3
        a = keras.layers.Input(inshp)
        b = keras.layers.ReLU(max_value=max_value,
                              negative_slope=negative_slope,
                              threshold=threshold)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ReLU_after_Dense(self):
        # Regression: advanced-activation layer fed by a non-input layer must
        # emit `tensor.array`, not `&tensor.array`, in the generated C call.
        a = keras.layers.Input((4,))
        b = keras.layers.Dense(8)(a)
        c = keras.layers.ReLU()(b)
        model = keras.models.Model(inputs=a, outputs=c)
        name = 'test___ReLU_after_Dense' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_selu(self):
        inshp = (8, 6, 5)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('selu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___SILU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_elu_activation(self):
        inshp = (10, 7, 4)
        a = keras.layers.Input(inshp)
        b = keras.layers.Dense(12, activation='elu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___elu_act' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_gelu(self):
        inshp = (7, 11, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('gelu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___gelu' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('hard_silu'), "Keras < 3.x lacks 'hard_silu'")
    def test_hard_silu(self):
        inshp = (6, 9, 4)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('hard_silu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___hard_silu' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_mish(self):
        inshp = (5, 8, 7)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('mish')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___mish' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_relu6(self):
        inshp = (10, 6, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('relu6')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___relu6' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_log_softmax(self):
        inshp = (8, 12)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('log_softmax')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___log_softmax' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_leaky_relu_activation(self):
        inshp = (7, 5, 9)
        a = keras.layers.Input(inshp)
        b = keras.layers.Dense(8, activation='leaky_relu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___leaky_relu_act' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('celu'), "Keras version lacks 'celu'")
    def test_celu(self):
        inshp = (6, 10, 4)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('celu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___celu' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('hard_tanh'), "Keras version lacks 'hard_tanh'")
    def test_hard_tanh(self):
        inshp = (9, 7, 5)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('hard_tanh')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___hard_tanh' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('hard_shrink'), "Keras version lacks 'hard_shrink'")
    def test_hard_shrink(self):
        inshp = (8, 6, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('hard_shrink')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___hard_shrink' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('soft_shrink'), "Keras version lacks 'soft_shrink'")
    def test_soft_shrink(self):
        inshp = (7, 5, 4)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('soft_shrink')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___soft_shrink' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('squareplus'), "Keras version lacks 'squareplus'")
    def test_squareplus(self):
        inshp = (5, 9, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('squareplus')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___squareplus' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    @unittest.skipUnless(_has_activation('sparse_plus'), "Keras version lacks 'sparse_plus'")
    def test_sparse_plus(self):
        inshp = (6, 8, 4)
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('sparse_plus')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___sparse_plus' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)
