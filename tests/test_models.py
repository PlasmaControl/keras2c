"""test_models.py
This file is part of the test suite for keras2c
Implements tests for full models
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


class TestModels(unittest.TestCase):
    """tests for full models"""

    def test_CIFAR_10_CNN(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(8, (3, 3), padding='same',
                                      input_shape=(32, 32, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(8, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(8, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(8, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(20))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Activation('softmax'))
        name = 'test___CIFAR_10_CNN' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ProfilePredictor(self):
        inshp = (8, 32)
        inp = keras.layers.Input(inshp)
        a = keras.layers.Dense(20, activation='relu')(inp)
        a = keras.layers.Dense(20, activation='relu')(a)
        a = keras.layers.LSTM(20, activation='relu')(a)
        outp = keras.layers.Dense(30)(a)
        model = keras.models.Model(inp, outp)
        name = 'test___ProfilePredictor' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    # def test_BabyMemNN(self):
    #     story_maxlen = 15
    #     query_maxlen = 22
    #     vocab_size = 50
    #     input_sequence = keras.layers.Input((story_maxlen,))
    #     question = keras.layers.Input((query_maxlen,))
    #     input_encoder_m = keras.models.Sequential()
    #     input_encoder_m.add(keras.layers.Embedding(input_dim=vocab_size,
    #                                                output_dim=64))
    #     input_encoder_m.add(keras.layers.Dropout(0.3))
    #     input_encoder_c = keras.models.Sequential()
    #     input_encoder_c.add(keras.layers.Embedding(input_dim=vocab_size,
    #                                                output_dim=query_maxlen))
    #     input_encoder_c.add(keras.layers.Dropout(0.3))
    #     question_encoder = keras.models.Sequential()
    #     question_encoder.add(keras.layers.Embedding(input_dim=vocab_size,
    #                                                 output_dim=64,
    #                                                 input_length=query_maxlen))
    #     question_encoder.add(keras.layers.Dropout(0.3))
    #     input_encoded_m = input_encoder_m(input_sequence)
    #     input_encoded_c = input_encoder_c(input_sequence)
    #     question_encoded = question_encoder(question)
    #     match = keras.layers.dot(
    #         [input_encoded_m, question_encoded], axes=(2, 2))
    #     match = keras.layers.Activation('softmax')(match)
    #     response = keras.layers.add([match, input_encoded_c])
    #     response = keras.layers.Permute((2, 1))(response)
    #     answer = keras.layers.concatenate([response, question_encoded])
    #     answer = keras.layers.LSTM(32)(answer)
    #     answer = keras.layers.Dropout(0.3)(answer)
    #     answer = keras.layers.Dense(vocab_size)(answer)
    #     answer = keras.layers.Activation('softmax')(answer)
    #     model = keras.models.Model([input_sequence, question], answer)
    #     name = 'test___BabyMemNN' + str(int(time.time()))
    #     keras2c_main.k2c(model, name)
    #     rcode = build_and_run(name)
    #     self.assertEqual(rcode, 0)


if __name__ == "__main__":
    unittest.main()
