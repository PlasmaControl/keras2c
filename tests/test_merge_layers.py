"""test_merge_layers.py
This file is part of the test suite for keras2c
Implements tests for merge layers
"""

#!/usr/bin/env python3

import unittest
import keras
from keras2c import keras2c_main
import subprocess
import time
import os
from test_core_layers import build_and_run

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)

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
        self.assertEqual(rcode, 0)


if __name__ == "__main__":
    unittest.main()
