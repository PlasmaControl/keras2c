"""test_optimizations.py
This file is part of the test suite for keras2c
Tests the codegen-time optimizations introduced in pcs_optimizations_static:
  - BatchNormalization folding into Dense / Conv1D
  - Non-mutation of the caller's model across a k2c() call
"""

#!/usr/bin/env python3

import os
import time
import unittest

import keras
import numpy as np
from keras import layers

from keras2c import keras2c_main
from keras2c.keras2c_main import fold_batch_norms


def _bn(scale=True, center=True, zero_offset=False):
    """BatchNormalization with reproducible non-trivial weights.

    zero_offset=True forces beta=0 and moving_mean=0 so that bn_offset==0,
    which is the regime where folding is mathematically safe even when the
    downstream conv pads its inputs with zeros.
    """
    init = keras.initializers.RandomUniform(minval=0.1, maxval=1.0)
    return layers.BatchNormalization(
        scale=scale,
        center=center,
        beta_initializer='zeros' if zero_offset else init,
        gamma_initializer=init,
        moving_mean_initializer='zeros' if zero_offset else init,
        moving_variance_initializer=init,
    )


def _build_bn_dense(input_dim=8, units=16, use_bias=True, scale=True, center=True,
                    zero_offset=False):
    inp = keras.Input((input_dim,))
    x = _bn(scale=scale, center=center, zero_offset=zero_offset)(inp)
    out = layers.Dense(units, use_bias=use_bias,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')(x)
    return keras.Model(inp, out)


def _build_bn_conv1d(time=20, in_channels=4, filters=8, kernel_size=3,
                     padding='valid', use_bias=True, scale=True, center=True,
                     zero_offset=False):
    inp = keras.Input((time, in_channels))
    x = _bn(scale=scale, center=center, zero_offset=zero_offset)(inp)
    out = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                        padding=padding, use_bias=use_bias,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros')(x)
    return keras.Model(inp, out)


def _clone(model):
    c = keras.models.clone_model(model)
    c.set_weights(model.get_weights())
    return c


class TestBatchNormFolding(unittest.TestCase):
    """Folding logic in keras2c_main.fold_batch_norms — Python-only, no C build."""

    def _assert_outputs_match(self, ref, folded, x, atol=1e-5):
        y1 = ref.predict(x, verbose=0)
        y2 = folded.predict(x, verbose=0)
        diff = float(np.max(np.abs(y1 - y2)))
        self.assertLess(diff, atol,
                        msg=f'output diverges after fold: max abs diff={diff:.3e}')

    def test_fold_bn_into_dense(self):
        model = _build_bn_dense()
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 1, 'BN before Dense should fold')
        x = np.random.randn(4, 8).astype(np.float32)
        self._assert_outputs_match(ref, model, x)

    def test_fold_bn_into_conv1d_valid(self):
        model = _build_bn_conv1d(padding='valid')
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 1, "BN before Conv1D padding='valid' should fold")
        x = np.random.randn(2, 20, 4).astype(np.float32)
        self._assert_outputs_match(ref, model, x)

    def test_fold_bn_into_conv1d_same_is_safe(self):
        # padding='same' pads edges with zeros, so a constant bn_offset does
        # not propagate uniformly through the kernel — folding is mathematically
        # unsafe at edge outputs. fold_batch_norms must skip this case so the
        # generated output stays exact.
        model = _build_bn_conv1d(padding='same')
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 0,
                         "BN before Conv1D padding='same' must not be folded")
        x = np.random.randn(2, 20, 4).astype(np.float32)
        self._assert_outputs_match(ref, model, x, atol=1e-6)

    def test_fold_bn_into_conv1d_causal_is_safe(self):
        model = _build_bn_conv1d(padding='causal')
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 0,
                         "BN before Conv1D padding='causal' must not be folded")
        x = np.random.randn(2, 20, 4).astype(np.float32)
        self._assert_outputs_match(ref, model, x, atol=1e-6)

    def test_fold_bn_into_conv1d_same_kernel1_ok(self):
        # kernel_size=1 has no real padding so 'same' folds correctly.
        model = _build_bn_conv1d(padding='same', kernel_size=1)
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 1)
        x = np.random.randn(2, 20, 4).astype(np.float32)
        self._assert_outputs_match(ref, model, x)

    def test_fold_bn_into_conv1d_same_no_offset_ok(self):
        # With beta=0 and moving_mean=0 the BN offset is zero, so the 'same'
        # padding edge issue disappears and folding is safe.
        model = _build_bn_conv1d(padding='same', zero_offset=True)
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 1,
                         'BN with no offset should fold even with same padding')
        x = np.random.randn(2, 20, 4).astype(np.float32)
        self._assert_outputs_match(ref, model, x)

    def test_fold_bn_into_dense_no_bias(self):
        # Dense has no bias; folding only applies when the produced offset is
        # zero. zero_offset=True (beta=0, mean=0) gives bn_offset=0.
        model = _build_bn_dense(use_bias=False, zero_offset=True)
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 1)
        x = np.random.randn(4, 8).astype(np.float32)
        self._assert_outputs_match(ref, model, x)

    def test_fold_skipped_for_dense_no_bias_with_offset(self):
        # No-bias Dense with non-zero BN offset has nowhere to absorb the
        # constant — fold_batch_norms must skip it.
        model = _build_bn_dense(use_bias=False)  # default initialisers => non-zero mean
        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 0,
                         'no-bias Dense with non-zero offset must not be folded')
        x = np.random.randn(4, 8).astype(np.float32)
        self._assert_outputs_match(ref, model, x, atol=1e-6)

    def test_fold_skipped_with_parallel_consumers(self):
        # BN output feeds two Dense layers in parallel — folding must be skipped
        # because absorbing BN into one consumer would break the other.
        inp = keras.Input((8,))
        x = _bn()(inp)
        a = layers.Dense(4, name='dense_a')(x)
        b = layers.Dense(4, name='dense_b')(x)
        out = layers.Concatenate()([a, b])
        model = keras.Model(inp, out)

        ref = _clone(model)
        folded = fold_batch_norms(model, verbose=False)
        self.assertEqual(len(folded), 0,
                         'BN with multiple consumers must not be folded')
        # And the model's behaviour must be identical (no weight changes).
        x_in = np.random.randn(4, 8).astype(np.float32)
        self._assert_outputs_match(ref, model, x_in, atol=1e-6)


class TestK2CDoesNotMutateModel(unittest.TestCase):
    """k2c must not change the caller's model — folding works on a clone."""

    def _cleanup(self, name):
        for ext in ('.c', '.h'):
            p = name + ext
            if os.path.exists(p):
                os.remove(p)

    def test_weights_and_outputs_unchanged_after_k2c(self):
        model = _build_bn_dense()
        x = np.random.randn(4, 8).astype(np.float32)
        y_before = model.predict(x, verbose=0)
        weights_before = [w.copy() for w in model.get_weights()]

        name = 'test___k2c_no_mutate' + str(int(time.time()))
        try:
            keras2c_main.k2c(model, name, num_tests=0, verbose=False)
            weights_after = model.get_weights()
            self.assertEqual(len(weights_before), len(weights_after))
            for wb, wa in zip(weights_before, weights_after):
                self.assertTrue(
                    np.array_equal(wb, wa),
                    msg=f'k2c mutated a weight (max abs diff={np.max(np.abs(wb-wa)):.3e})',
                )
            y_after = model.predict(x, verbose=0)
            self.assertTrue(np.array_equal(y_before, y_after),
                            msg='k2c changed model output')
        finally:
            self._cleanup(name)


if __name__ == '__main__':
    unittest.main()
