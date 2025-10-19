#!/usr/bin/env python3

import unittest
import tensorflow.keras as keras
from keras2c import keras2c_main
import time
import numpy as np
from test_core_layers import build_and_run
import tensorflow as tf
from keras2c.io_parsing import get_layer_io_names
tf.compat.v1.disable_eager_execution()

class TestSplitLayers(unittest.TestCase):
    """tests for split layers"""

    def test_split_layers(self):
        for tno in range (10):
            n_splits = np.random.randint(2, 10)
            in_dim = np.random.randint(1 * n_splits, 10 * n_splits)
            split_sizes = []
            for i in range(n_splits - 1):
                split_sizes.append(np.random.randint(1, in_dim - sum(split_sizes) - n_splits + i + 2))
            split_sizes.append(in_dim - sum(split_sizes))
            a = keras.layers.Input(in_dim)
            b = tf.split(a, split_sizes, axis=1)
            model = keras.models.Model(inputs=a, outputs=b)
            name = 'test___SplitLayer' + str(int(time.time()))
            keras2c_main.k2c(model, name)
            rcode = build_and_run(name)
            self.assertEqual(rcode, 0)

if __name__ == "__main__":
    unittest.main()
