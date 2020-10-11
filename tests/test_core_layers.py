"""test_core_layers.py
This file is part of the test suite for keras2c
Implements tests for core layers
"""

#!/usr/bin/env python3

import unittest
import tensorflow.keras as keras
from keras2c import keras2c_main
import subprocess
import time
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


CC = 'gcc'


def build_and_run(name, return_output=False):

    cwd = os.getcwd()
    os.chdir(os.path.abspath('./include/'))
    lib_code = subprocess.run(['make']).returncode
    os.chdir(os.path.abspath(cwd))
    if lib_code != 0:
        return 'lib build failed'

    if os.environ.get('CI'):
        ccflags = '-g -Og -std=c99 --coverage -I./include/'
    else:
        ccflags = '-Ofast -std=c99 -I./include/'

    cc = CC + ' ' + ccflags + ' -o ' + name + ' ' + name + '.c ' + \
        name + '_test_suite.c -L./include/ -l:libkeras2c.a -lm'
    build_code = subprocess.run(cc.split()).returncode
    if build_code != 0:
        return 'build failed'
    proc_output = subprocess.run(['./' + name])
    rcode = proc_output.returncode
    if rcode == 0:
        if not os.environ.get('CI'):
            subprocess.run('rm ' + name + '*', shell=True)
            return (rcode, proc_output.stdout) if return_output else rcode
    return rcode


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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

    def test_Permute(self):
        inshp = (6, 12, 9)
        a = keras.layers.Input(inshp)
        b = keras.layers.Permute((3, 1, 2))(a)
     #   c = keras.layers.Dense(20)(b)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___permute' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)


class TestEmbedding(unittest.TestCase):
    """tests for embedding layers"""

    def test_Embedding1(self):
        inshp = (10, 20)
        input_dim = 20
        output_dim = 30
        a = keras.layers.Input(inshp)
        b = keras.layers.Activation('relu')(a)
        c = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)(b)
        model = keras.models.Model(inputs=a, outputs=c)
        name = 'test___Embedding1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


class TestNormalization(unittest.TestCase):
    """tests for normalization layers"""

    def test_BatchNorm1(self):
        inshp = (10, 11, 12)
        axis = 3
        init = keras.initializers.RandomUniform(minval=0.1, maxval=1.0)
        a = keras.layers.Input(inshp)
        b = keras.layers.BatchNormalization(axis=axis,
                                            beta_initializer=init,
                                            gamma_initializer=init,
                                            moving_mean_initializer=init,
                                            moving_variance_initializer=init,
                                            scale=True, center=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___BatchNorm1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_BatchNorm2(self):
        inshp = (10, 11, 12)
        axis = 2
        init = keras.initializers.RandomUniform(minval=0.1, maxval=1.0)
        a = keras.layers.Input(inshp)
        b = keras.layers.BatchNormalization(axis=axis,
                                            beta_initializer=init,
                                            gamma_initializer=init,
                                            moving_mean_initializer=init,
                                            moving_variance_initializer=init,
                                            scale=False, center=True)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___BatchNorm2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_BatchNorm3(self):
        inshp = (10, 11, 12, 13)
        axis = 1
        init = keras.initializers.RandomUniform(minval=0.1, maxval=1.0)
        a = keras.layers.Input(inshp)
        b = keras.layers.BatchNormalization(axis=axis,
                                            beta_initializer=init,
                                            gamma_initializer=init,
                                            moving_mean_initializer=init,
                                            moving_variance_initializer=init,
                                            scale=True, center=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___BatchNorm3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_BatchNorm4(self):
        inshp = (10, 11, 12)
        axis = 2
        init = keras.initializers.RandomUniform(minval=0.1, maxval=2.0)
        a = keras.layers.Input(inshp)
        b = keras.layers.BatchNormalization(axis=axis,
                                            beta_initializer=init,
                                            gamma_initializer=init,
                                            moving_mean_initializer=init,
                                            moving_variance_initializer=init,
                                            scale=False, center=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___BatchNorm4' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


class TestSharedLayers(unittest.TestCase):
    """tests for shared layers"""

    def test_SharedLayer1(self):
        inshp = (10, 20)
        xi = keras.layers.Input(inshp)
        x = keras.layers.Dense(20, activation='relu')(xi)
        yi = keras.layers.Input(inshp)
        y = keras.layers.Dense(20, activation='relu')(yi)
        f = keras.layers.Dense(30, activation='relu')
        x = f(x)
        y = f(y)
        z = keras.layers.Add()([x, y])
        model = keras.models.Model(inputs=[xi, yi], outputs=z)
        name = 'test___SharedLayer1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


if __name__ == "__main__":
    unittest.main()
