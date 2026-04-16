"""Backend utilities for :mod:`keras2c`.

This module exposes the Keras API used throughout the project. By keeping the
backend import in a single place it becomes easier to switch to a different
deep learning framework in the future.  Currently TensorFlow's Keras
implementation is used which requires TensorFlow 2.x.
"""

from tensorflow import keras

__all__ = ["keras"]

