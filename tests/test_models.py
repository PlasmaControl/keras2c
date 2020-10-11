"""test_models.py
This file is part of the test suite for keras2c
Implements tests for full models
"""

#!/usr/bin/env python3

import unittest
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Conv2D, ConvLSTM2D, Dot, Add, Multiply, Concatenate, Reshape, Permute, ZeroPadding1D, Cropping1D
from tensorflow.keras.models import Model
import numpy as np
from keras2c import keras2c_main
import subprocess
import time
import os
from test_core_layers import build_and_run
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
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

    def test_ProfilePredictorConv1D(self):
        input_profile_names = ['a', 'b', 'c']
        target_profile_names = ['a', 'b']
        actuator_names = ['aa', 'bb', 'cc']
        lookbacks = {'a': 1,
                     'b': 1,
                     'c': 1,
                     'aa': 5,
                     'bb': 5,
                     'cc': 5}
        lookahead = 4
        profile_length = 33
        std_activation = 'relu'

        rnn_layer = layers.LSTM

        profile_inshape = (lookbacks[input_profile_names[0]], profile_length)
        past_actuator_inshape = (lookbacks[actuator_names[0]],)
        future_actuator_inshape = (lookahead,)
        num_profiles = len(input_profile_names)
        num_targets = len(target_profile_names)
        num_actuators = len(actuator_names)

        # input each profile sig one by one and then concat them together
        profile_inputs = []
        profiles = []
        for i in range(num_profiles):
            profile_inputs.append(
                Input(profile_inshape, name='input_' + input_profile_names[i]))
            profiles.append(Reshape((lookbacks[input_profile_names[i]], profile_length, 1))
                            (profile_inputs[i]))
        current_profiles = Concatenate(axis=-1)(profiles)
        current_profiles = Reshape(
            (profile_length, num_profiles))(current_profiles)

        # input previous and future actuators and concat each of them
        actuator_past_inputs = []
        actuator_future_inputs = []

        previous_actuators = []
        future_actuators = []

        for i in range(num_actuators):
            actuator_future_inputs.append(
                Input(future_actuator_inshape,
                      name="input_future_{}".format(actuator_names[i]))
            )
            actuator_past_inputs.append(
                Input(past_actuator_inshape,
                      name="input_past_{}".format(actuator_names[i]))
            )

            future_actuators.append(Reshape((lookahead, 1))
                                    (actuator_future_inputs[i]))
            previous_actuators.append(
                Reshape((lookbacks[actuator_names[i]], 1))(actuator_past_inputs[i]))

        future_actuators = Concatenate(axis=-1)(future_actuators)
        previous_actuators = Concatenate(axis=-1)(previous_actuators)

        print(future_actuators.shape)
        print(previous_actuators.shape)
        print(current_profiles.shape)

        #######################################################################

        actuator_effect = rnn_layer(
            profile_length, activation=std_activation)(previous_actuators)
        actuator_effect = layers.Reshape(
            target_shape=(profile_length, 1))(actuator_effect)

        future_actuator_effect = rnn_layer(
            profile_length, activation=std_activation)(future_actuators)
        future_actuator_effect = layers.Reshape(
            target_shape=(profile_length, 1))(future_actuator_effect)

        current_profiles_processed_0 = layers.Concatenate()(
            [current_profiles, actuator_effect, future_actuator_effect])

        prof_act = []
        for i in range(num_targets):

            current_profiles_processed_1 = layers.Conv1D(filters=8, kernel_size=2,
                                                         padding='same',
                                                         activation='relu')(current_profiles_processed_0)
            current_profiles_processed_2 = layers.Conv1D(filters=8, kernel_size=4,
                                                         padding='same',
                                                         activation='relu')(current_profiles_processed_1)
            current_profiles_processed_3 = layers.Conv1D(filters=8, kernel_size=8,
                                                         padding='same',
                                                         activation='relu')(current_profiles_processed_2)

            final_output = layers.Concatenate()(
                [current_profiles_processed_1, current_profiles_processed_2, current_profiles_processed_3])
            final_output = layers.Conv1D(filters=10, kernel_size=4,
                                         padding='same', activation='tanh')(final_output)
            final_output = layers.Conv1D(filters=1, kernel_size=4,
                                         padding='same', activation='linear')(final_output)
            final_output = layers.Reshape(target_shape=(
                profile_length,), name="target_"+target_profile_names[i])(final_output)

            prof_act.append(final_output)
        print(len(prof_act))

        model = Model(inputs=profile_inputs + actuator_past_inputs +
                      actuator_future_inputs, outputs=prof_act)

        name = 'test___ProfilePredictorConv1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ProfilePredictorConv2D(self):
        input_profile_names = ['a', 'b', 'c']
        target_profile_names = ['a', 'b']
        actuator_names = ['aa', 'bb', 'cc']
        lookbacks = {'a': 1,
                     'b': 1,
                     'c': 1,
                     'aa': 5,
                     'bb': 5,
                     'cc': 5}
        profile_lookback = 1
        actuator_lookback = 5
        lookahead = 4
        profile_length = 33
        std_activation = 'relu'

        profile_inshape = (profile_lookback, profile_length)
        past_actuator_inshape = (actuator_lookback,)
        future_actuator_inshape = (lookahead,)
        num_profiles = len(input_profile_names)
        num_targets = len(target_profile_names)
        num_actuators = len(actuator_names)
        max_channels = 32

        profile_inputs = []
        profiles = []
        for i in range(num_profiles):
            profile_inputs.append(
                Input(profile_inshape, name='input_' + input_profile_names[i]))
            profiles.append(Reshape((profile_lookback, profile_length, 1))
                            (profile_inputs[i]))
        profiles = Concatenate(axis=-1)(profiles)
        # shape = (lookback, length, channels=num_profiles)
        profiles = Conv2D(filters=int(num_profiles*max_channels/8),
                          kernel_size=(1, int(profile_length/12)),
                          strides=(1, 1), padding='same', activation=std_activation)(profiles)
        profiles = Conv2D(filters=int(num_profiles*max_channels/4),
                          kernel_size=(1, int(profile_length/8)),
                          strides=(1, 1), padding='same', activation=std_activation)(profiles)
        profiles = Conv2D(filters=int(num_profiles*max_channels/2),
                          kernel_size=(1, int(profile_length/6)),
                          strides=(1, 1), padding='same', activation=std_activation)(profiles)
        profiles = Conv2D(filters=int(num_profiles*max_channels),
                          kernel_size=(1, int(profile_length/4)),
                          strides=(1, 1), padding='same', activation=std_activation)(profiles)
        # shape = (lookback, length, channels)
        if profile_lookback > 1:
            profiles = Conv2D(filters=int(num_profiles*max_channels), kernel_size=(profile_lookback, 1),
                              strides=(1, 1), padding='valid', activation=std_activation)(profiles)
        profiles = Reshape((profile_length, int(
            num_profiles*max_channels)))(profiles)
        # shape = (length, channels)

        actuator_future_inputs = []
        actuator_past_inputs = []
        actuators = []
        for i in range(num_actuators):
            actuator_future_inputs.append(
                Input(future_actuator_inshape, name='input_future_' + actuator_names[i]))
            actuator_past_inputs.append(
                Input(past_actuator_inshape, name='input_past_' + actuator_names[i]))
            actuators.append(Concatenate(
                axis=-1)([actuator_past_inputs[i], actuator_future_inputs[i]]))
            actuators[i] = Reshape(
                (actuator_lookback+lookahead, 1))(actuators[i])
        actuators = Concatenate(axis=-1)(actuators)
        # shaoe = (time, num_actuators)
        actuators = Dense(units=int(num_profiles*max_channels/8),
                          activation=std_activation)(actuators)
        # actuators = Conv1D(filters=int(num_profiles*max_channels/8), kernel_size=3, strides=1,
        #                    padding='causal', activation=std_activation)(actuators)
        actuators = Dense(units=int(num_profiles*max_channels/4),
                          activation=std_activation)(actuators)
        # actuators = Conv1D(filters=int(num_profiles*max_channels/4), kernel_size=3, strides=1,
        #                    padding='causal', activation=std_activation)(actuators)
        actuators = Dense(units=int(num_profiles*max_channels/2),
                          activation=std_activation)(actuators)
        actuators = LSTM(units=int(num_profiles*max_channels), activation=std_activation,
                         recurrent_activation='hard_sigmoid')(actuators)
        actuators = Reshape((int(num_profiles*max_channels), 1))(actuators)
        # shape = (channels, 1)
        actuators = Dense(units=int(profile_length/4),
                          activation=std_activation)(actuators)
        actuators = Dense(units=int(profile_length/2),
                          activation=std_activation)(actuators)
        actuators = Dense(units=profile_length, activation=None)(actuators)
        # shape = (channels, profile_length)
        actuators = Permute(dims=(2, 1))(actuators)
        # shape = (profile_length, channels)

        merged = Add()([profiles, actuators])
        merged = Reshape((1, profile_length, int(
            num_profiles*max_channels)))(merged)
        # shape = (1, length, channels)

        prof_act = []
        for i in range(num_targets):
            prof_act.append(Conv2D(filters=max_channels, kernel_size=(1, int(profile_length/4)), strides=(1, 1),
                                   padding='same', activation=std_activation)(merged))
            # shape = (1,length,max_channels)
            prof_act[i] = Conv2D(filters=int(max_channels/2), kernel_size=(1, int(profile_length/8)),
                                 strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
            prof_act[i] = Conv2D(filters=int(max_channels/4), kernel_size=(1, int(profile_length/6)),
                                 strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
            prof_act[i] = Conv2D(filters=int(max_channels/8), kernel_size=(1, int(profile_length/4)),
                                 strides=(1, 1), padding='same', activation=std_activation)(prof_act[i])
            prof_act[i] = Conv2D(filters=1, kernel_size=(1, int(profile_length/4)), strides=(1, 1),
                                 padding='same', activation=None)(prof_act[i])
            # shape = (1,length,1)
            prof_act[i] = Reshape((profile_length,), name='target_' +
                                  target_profile_names[i])(prof_act[i])
        model = Model(inputs=profile_inputs + actuator_past_inputs +
                      actuator_future_inputs, outputs=prof_act)
        name = 'test___ProfilePredictorConv2D' + str(int(time.time()))
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
