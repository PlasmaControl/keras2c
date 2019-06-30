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


class TestAdvancedActivation(unittest.TestCase):
    """tests for advanced activation layers"""

    def test_LeakyReLU(self):
        inshp = (9, 7, 6, 3)
        alpha = 0.5
        a = keras.layers.Input(inshp)
        b = keras.layers.LeakyReLU(alpha=alpha)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LeakyReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_PReLU(self):
        inshp = (3, 6, 9, 3)
        a = keras.layers.Input(inshp)
        b = keras.layers.PReLU(alpha_initializer='glorot_uniform')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___PReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_ELU(self):
        inshp = (13, 6, 9, 13)
        alpha = 1.3
        a = keras.layers.Input(inshp)
        b = keras.layers.ELU(alpha=alpha)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ELU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_ThresholdedReLU(self):
        inshp = (3, 6, 19, 11)
        theta = 0.3
        a = keras.layers.Input(inshp)
        b = keras.layers.ThresholdedReLU(theta=theta)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___ThresholdedReLU' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)


class TestPoolingLayers(unittest.TestCase):
    """tests for pooling layers"""

    def test_MaxPooling1D1(self):
        inshp = (23, 29)
        pool_size = 3
        strides = 1
        padding = 'valid'
        a = keras.layers.Input(inshp)
        b = keras.layers.MaxPooling1D(pool_size=pool_size,
                                      strides=strides,
                                      padding=padding)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___MaxPooling1D1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)

    def test_GlobalAveragePooling1D(self):
        inshp = (16, 11)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalAveragePooling1D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalAveragePooling1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_GlobalMaxPooling1D(self):
        inshp = (31, 21)
        a = keras.layers.Input(inshp)
        b = keras.layers.GlobalMaxPooling1D()(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GlobalMaxPooling1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)


class TestMergeLayers(unittest.TestCase):
    """tests for merge layers"""

    def test_Dot1(self):
        inshp1 = (10, 8)
        inshp2 = (8, 12)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Dot((2, 1))([a, b])
        model = keras.models.Model(inputs=[a, b], outputs=c)
        name = 'test___Dot1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Add1(self):
        inshp1 = (10, 8, 12)
        inshp2 = (10, 8, 12)
        inshp3 = (10, 8, 12)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Input(inshp3)
        d = keras.layers.Add()([a, b, c])
        model = keras.models.Model(inputs=[a, b, c], outputs=d)
        name = 'test___Add1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Subtract1(self):
        inshp1 = (10, 8, 2, 3)
        inshp2 = (10, 8, 2, 3)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Subtract()([a, b])
        model = keras.models.Model(inputs=[a, b], outputs=c)
        name = 'test___Subtract1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Multiply1(self):
        inshp1 = (9, 12)
        inshp2 = (9, 12)
        inshp3 = (9, 12)
        inshp4 = (9, 12)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Input(inshp3)
        d = keras.layers.Input(inshp4)
        e = keras.layers.Multiply()([a, b, c, d])
        model = keras.models.Model(inputs=[a, b, c, d], outputs=e)
        name = 'test___Multiply1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Maximum1(self):
        inshp1 = (9, 14)
        inshp2 = (9, 14)
        inshp3 = (9, 14)
        inshp4 = (9, 14)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Input(inshp3)
        d = keras.layers.Input(inshp4)
        e = keras.layers.Maximum()([a, b, c, d])
        model = keras.models.Model(inputs=[a, b, c, d], outputs=e)
        name = 'test___Maximum1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Minimum1(self):
        inshp1 = (7, 12, 2)
        inshp2 = (7, 12, 2)
        inshp3 = (7, 12, 2)
        inshp4 = (7, 12, 2)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Input(inshp3)
        d = keras.layers.Input(inshp4)
        e = keras.layers.Minimum()([a, b, c, d])
        model = keras.models.Model(inputs=[a, b, c, d], outputs=e)
        name = 'test___Minimum1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Average1(self):
        inshp1 = (4, 12, 2)
        inshp2 = (4, 12, 2)
        inshp3 = (4, 12, 2)
        inshp4 = (4, 12, 2)
        a = keras.layers.Input(inshp1)
        b = keras.layers.Input(inshp2)
        c = keras.layers.Input(inshp3)
        d = keras.layers.Input(inshp4)
        e = keras.layers.Average()([a, b, c, d])
        model = keras.models.Model(inputs=[a, b, c, d], outputs=e)
        name = 'test___Average1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)


class TestConvolutionLayers(unittest.TestCase):
    """tests for convolution layers"""

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
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)

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
        self.assertEqual(rcode.returncode, 0)


class TestCoreLayers(unittest.TestCase):
    """tests for core layers"""

    def test_Dense1(self):
        inshp = (21, 4, 9)
        units = 45
        a = keras.layers.Input(inshp)
        b = keras.layers.Dense(units, activation='relu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Dense1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Dense2_Activation(self):
        inshp = (40, 30)
        units = 500
        a = keras.layers.Input(inshp)
        b = keras.layers.Dense(units, activation='tanh', use_bias=False)(a)
        c = keras.layers.Activation('exponential')(b)
        model = keras.models.Model(inputs=a, outputs=c)
        name = 'test___Dense2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Dropout_Reshape_Flatten(self):
        inshp = (10, 40, 30)
        a = keras.layers.Input(inshp)
        b = keras.layers.Flatten()(a)
        c = keras.layers.Dropout(.4)(b)
        d = keras.layers.Reshape((20, 30, 20))(c)
        model = keras.models.Model(inputs=a, outputs=d)
        name = 'test___flatten_dropout_reshape' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_Permute(self):
        inshp = (6, 12, 9)
        a = keras.layers.Input(inshp)
        b = keras.layers.Permute((3, 1, 2))(a)
     #   c = keras.layers.Dense(20)(b)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___permute' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_repeat_vector(self):
        inshp = (13,)
        a = keras.layers.Input(inshp)
        b = keras.layers.RepeatVector(23)(a)
        c = keras.layers.ActivityRegularization(l1=.5, l2=.3)(b)
        d = keras.layers.Dense(20)(c)
        model = keras.models.Model(inputs=a, outputs=d)
        name = 'test___repeat_vector' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_dummy_layers(self):
        inshp = (4, 5, 6, 7)
        a = keras.layers.Input(inshp)
        b = keras.layers.SpatialDropout3D(.2)(a)
        c = keras.layers.Reshape((20, 6, 7))(b)
        d = keras.layers.SpatialDropout2D(.3)(c)
        e = keras.layers.Reshape((20, 42))(d)
        f = keras.layers.SpatialDropout1D(.4)(e)
        g = keras.layers.Flatten()(f)
        model = keras.models.Model(inputs=a, outputs=g)
        name = 'test___dummy_layers' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)


class TestRecurrentLayers(unittest.TestCase):
    """tests for recurrent layers"""

    def test_SimpleRNN1(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units, activation='relu',
                                   return_sequences=False)(a)
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

    def test_SimpleRNN3(self):
        inshp = (34, 17)
        units = 40
        a = keras.layers.Input(inshp)
        b = keras.layers.SimpleRNN(units, go_backwards=False,
                                   return_sequences=True,
                                   activation='tanh',
                                   use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___SimpleRNN3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_LSTM1(self):
        inshp = (23, 32)
        units = 19
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(units, activation='relu',
                              return_sequences=False,
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

    def test_LSTM3(self):
        inshp = (4, 80)
        units = 23
        a = keras.layers.Input(inshp)
        b = keras.layers.LSTM(units, go_backwards=False,
                              return_sequences=True,
                              activation='sigmoid',
                              recurrent_activation='tanh',
                              use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___LSTM3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)

    def test_GRU1(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units, activation='softmax',
                             recurrent_activation='softsign',
                             return_sequences=False)(a)
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

    def test_GRU3(self):
        inshp = (12, 46)
        units = 17
        a = keras.layers.Input(inshp)
        b = keras.layers.GRU(units, activation='softplus',
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             go_backwards=False,
                             reset_after=True,
                             use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___GRU3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode.returncode, 0)


if __name__ == "__main__":
    unittest.main()
