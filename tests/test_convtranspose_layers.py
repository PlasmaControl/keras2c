"""test_convolution_layers.py
This file is part of the test suite for keras2c
Implements tests for convolution layers
"""

#!/usr/bin/env python3

import unittest
import tensorflow.keras as keras
from keras2c import keras2c_main
import time
from test_core_layers import build_and_run
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

__author__ = "Anchal Gupta"
__copyright__ = "Copyright 2024, Anchal Gupta"
__license__ = "MIT"
__maintainer__ = "Anchal Gupta, https://github.com/anchal-physics/keras2c"
__email__ = "guptaa@fusion.gat.com"


class TestConvolutionTransposeLayers(unittest.TestCase):
    """tests for convolution layers"""

    def test_Conv1DTranspose1(self):
        for tno in range (10):
            nh = np.random.randint(2, 50)
            nc = np.random.randint(1, 50)
            nf = np.random.randint(1, 50)
            nk = np.random.randint(1, nh)
            strides = np.random.randint(1, max(nk, 2))
            inshp = (nh, nc)
            if tno % 2 == 0:
                padding = 'valid'
            else:
                padding = 'same'
            dilation_rate = 1
            activation = None # 'relu'
            a = keras.layers.Input(inshp)
            b = keras.layers.Conv1DTranspose(filters=nf,
                                            kernel_size=nk,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            activation=activation,
                                            use_bias=False)(a)
            model = keras.models.Model(inputs=a, outputs=b)
            name = 'test___Conv1DTranspose1' + str(int(time.time()))
            keras2c_main.k2c(model, name)
            rcode = build_and_run(name)
            self.assertEqual(rcode, 0)

if __name__ == "__main__":
    unittest.main()
