#!/usr/bin/env python3

import unittest
import keras
from keras2c import keras2c_main
import time
import numpy as np
from test_core_layers import build_and_run
import tensorflow as tf


class TestSplitLayers(unittest.TestCase):
    """tests for split layers"""

    def test_split_layers(self):
        for tno in range(10):
            n_splits = np.random.randint(2, 10)
            in_dim = np.random.randint(1 * n_splits, 10 * n_splits)

            # Method 1: Random distribution that guarantees validity
            split_sizes = [1] * n_splits  # Start with minimum size of 1 for each split
            remaining = in_dim - n_splits  # Distribute the remaining dimensions

            # Randomly distribute the remaining dimensions
            for _ in range(remaining):
                idx = np.random.randint(0, n_splits)
                split_sizes[idx] += 1

            # Convert to list of integers (sometimes numpy types cause issues)
            split_sizes = [int(s) for s in split_sizes]

            # Create model with these pre-computed split sizes
            a = keras.layers.Input((in_dim,))

            cumsum = np.cumsum(split_sizes[:-1]).tolist()  # Don't include the last one
            b = keras.ops.split(a, indices_or_sections=cumsum, axis=1)

            model = keras.models.Model(inputs=a, outputs=b)
            name = 'test___SplitLayer' + str(int(time.time()))
            keras2c_main.k2c(model, name)
            rcode = build_and_run(name)
            self.assertEqual(rcode, 0)
