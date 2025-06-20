"""test_core_layers.py
This file is part of the test suite for keras2c
Implements tests for core layers
"""

#!/usr/bin/env python3

import unittest
import os
from tensorflow import keras
from keras2c import keras2c_main
import subprocess
import time

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


CC = os.environ.get('CC', 'gcc')


def build_and_run(name, return_output=False):
    cwd = os.getcwd()
    os.chdir(os.path.abspath('./include/'))
    lib_process = subprocess.run(
        'make CC=' + CC,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    os.chdir(os.path.abspath(cwd))
    if lib_process.returncode != 0:
        print("Library build failed with the following output:")
        print(lib_process.stdout)
        print(lib_process.stderr)
        return 'lib build failed'

    if os.environ.get('CI'):
        ccflags = '-g -Og -std=c99 --coverage -I./include/'
    else:
        ccflags = '-Ofast -std=c99 -I./include/'

    cc = (
        CC
        + ' '
        + ccflags
        + ' -o '
        + name
        + ' '
        + name
        + '.c '
        + name
        + '_test_suite.c -L./include/ -lkeras2c -lm'
    )
    build_process = subprocess.run(
        cc,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if build_process.returncode != 0:
        print("Compilation failed with the following output:")
        print(build_process.stdout)
        print(build_process.stderr)
        return 'compilation failed'
    proc_output = subprocess.run(
        ['./' + name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    rcode = proc_output.returncode
    if not os.environ.get('CI'):
        subprocess.run(
            'rm ' + name + '*',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    return (rcode, proc_output.stdout) if return_output else rcode


class TestCoreLayers(unittest.TestCase):
    """
    Unit tests for core Keras layers using keras2c.
    This test suite includes the following tests:
    - test_Dense1: Tests a Dense layer with ReLU activation.
    - test_Dense2_Activation: Tests a Dense layer without bias followed by an
      Activation layer with exponential activation.
    - test_Dropout_Reshape_Flatten: Tests a sequence of Flatten, Dropout, and Reshape layers.
    - test_Permute: Tests a Permute layer.
    - test_repeat_vector: Tests a RepeatVector layer followed by an ActivityRegularization and Dense layer.
    - test_dummy_layers: Tests a sequence of SpatialDropout3D, Reshape,
      SpatialDropout2D, Reshape, SpatialDropout1D, and Flatten layers.
    Each test builds a Keras model, converts it using keras2c, and verifies that the generated code runs successfully.
    """


    def test_Activation1(self):
        inshp = (10, 20)
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Activation('relu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Activation1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Dense1(self):
        inshp = (21, 4, 9)
        units = 45
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Dense(units, activation='relu')(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Dense1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Dense2_NoBias(self):
        inshp = (40, 30)
        units = 500
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Dense(units, activation='tanh', use_bias=False)(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___Dense2' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Dense3_Activation(self):
        inshp = (40, 30)
        units = 500
        a = keras.layers.Input(shape=inshp, name='input_layer')
        b = keras.layers.Dense(units, activation='tanh', use_bias=False, name='dense_layer')(a)
        c = keras.layers.Activation('exponential', name='activation_layer')(b)
        model = keras.models.Model(inputs=a, outputs=c)
        name = 'test___Dense3' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_Dropout_Reshape_Flatten(self):
        inshp = (10, 40, 30)
        a = keras.layers.Input(shape=inshp)
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
        a = keras.layers.Input(shape=inshp)
        b = keras.layers.Permute((3, 1, 2))(a)
        model = keras.models.Model(inputs=a, outputs=b)
        name = 'test___permute' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_repeat_vector(self):
        inshp = (13,)
        a = keras.layers.Input(shape=inshp)
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
        a = keras.layers.Input(shape=inshp)
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
        inshp = (10,)
        input_dim = 50
        output_dim = 30
        a = keras.layers.Input(shape=inshp, dtype='int32')
        c = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)(a)
        model = keras.models.Model(inputs=a, outputs=c)
        name = 'test___Embedding1' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)


class TestNormalization(unittest.TestCase):
    """tests for normalization layers"""

    def test_BatchNorm1(self):
        inshp = (10, 11, 12)
        axis = -1
        init = keras.initializers.RandomUniform(minval=0.1, maxval=1.0)
        a = keras.layers.Input(shape=inshp)
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
        a = keras.layers.Input(shape=inshp)
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
        a = keras.layers.Input(shape=inshp)
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
        a = keras.layers.Input(shape=inshp)
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

    @unittest.skip  # no reason needed
    def test_SharedLayer1(self):
        inshp = (10, 20)
        xi = keras.layers.Input(shape=inshp)
        x = keras.layers.Dense(20, activation='relu')(xi)
        yi = keras.layers.Input(shape=inshp)
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
