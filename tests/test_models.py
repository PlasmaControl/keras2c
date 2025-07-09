"""test_models.py
This file is part of the test suite for keras2c
Implements tests for full models
"""

#!/usr/bin/env python3

import unittest
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, Conv2D, Add, Concatenate, Reshape,
    Permute, MaxPooling2D, Dropout, Flatten
)
from tensorflow.keras.models import Model, Sequential
from keras2c import keras2c_main
import time
from test_core_layers import build_and_run

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class TestModels(unittest.TestCase):
    """Tests for full models"""

    @unittest.skip  # no reason needed
    def test_CIFAR_10_CNN(self):
        model = Sequential()
        model.add(Conv2D(8, (3, 3), padding='same',
                         input_shape=(32, 32, 3), activation='relu'))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        name = 'test___CIFAR_10_CNN' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ProfilePredictor(self):
        inshp = (8, 32)
        inp = Input(shape=inshp)
        a = Dense(20, activation='relu')(inp)
        a = Dense(20, activation='relu')(a)
        a = LSTM(20, activation='relu')(a)
        outp = Dense(30)(a)
        model = Model(inputs=inp, outputs=outp)
        name = 'test___ProfilePredictor' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ProfilePredictorConv1D(self):
        input_profile_names = ['a', 'b', 'c']
        target_profile_names = ['a', 'b']
        actuator_names = ['aa', 'bb', 'cc']
        lookbacks = {'a': 1, 'b': 1, 'c': 1, 'aa': 5, 'bb': 5, 'cc': 5}
        lookahead = 4
        profile_length = 33
        std_activation = 'relu'

        rnn_layer = LSTM

        profile_inshape = (lookbacks[input_profile_names[0]], profile_length)
        past_actuator_inshape = (lookbacks[actuator_names[0]],)
        future_actuator_inshape = (lookahead,)
        num_profiles = len(input_profile_names)
        num_targets = len(target_profile_names)
        num_actuators = len(actuator_names)

        # Input each profile signal one by one and then concatenate them together
        profile_inputs = []
        profiles = []
        for i in range(num_profiles):
            profile_inputs.append(
                Input(shape=profile_inshape, name='input_' + input_profile_names[i])
            )
            profiles.append(
                Reshape((lookbacks[input_profile_names[i]], profile_length, 1))(profile_inputs[i])
            )
        current_profiles = Concatenate(axis=-1)(profiles)
        current_profiles = Reshape((profile_length, num_profiles))(current_profiles)

        # Input previous and future actuators and concatenate each of them
        actuator_past_inputs = []
        actuator_future_inputs = []
        previous_actuators = []
        future_actuators = []

        for i in range(num_actuators):
            actuator_future_inputs.append(
                Input(shape=future_actuator_inshape, name=f"input_future_{actuator_names[i]}")
            )
            actuator_past_inputs.append(
                Input(shape=past_actuator_inshape, name=f"input_past_{actuator_names[i]}")
            )

            future_actuators.append(
                Reshape((lookahead, 1))(actuator_future_inputs[i])
            )
            previous_actuators.append(
                Reshape((lookbacks[actuator_names[i]], 1))(actuator_past_inputs[i])
            )

        future_actuators = Concatenate(axis=-1)(future_actuators)
        previous_actuators = Concatenate(axis=-1)(previous_actuators)

        # Update recurrent_activation to 'sigmoid' as 'hard_sigmoid' is deprecated
        actuator_effect = rnn_layer(
            profile_length, activation=std_activation, recurrent_activation='sigmoid'
        )(previous_actuators)
        actuator_effect = Reshape(target_shape=(profile_length, 1))(actuator_effect)

        future_actuator_effect = rnn_layer(
            profile_length, activation=std_activation, recurrent_activation='sigmoid'
        )(future_actuators)
        future_actuator_effect = Reshape(target_shape=(profile_length, 1))(future_actuator_effect)

        current_profiles_processed_0 = Concatenate()(
            [current_profiles, actuator_effect, future_actuator_effect]
        )

        prof_act = []
        for i in range(num_targets):
            current_profiles_processed_1 = Conv1D(
                filters=8, kernel_size=2, padding='same', activation='relu'
            )(current_profiles_processed_0)
            current_profiles_processed_2 = Conv1D(
                filters=8, kernel_size=4, padding='same', activation='relu'
            )(current_profiles_processed_1)
            current_profiles_processed_3 = Conv1D(
                filters=8, kernel_size=8, padding='same', activation='relu'
            )(current_profiles_processed_2)

            final_output = Concatenate()(
                [
                    current_profiles_processed_1,
                    current_profiles_processed_2,
                    current_profiles_processed_3,
                ]
            )
            final_output = Conv1D(
                filters=10, kernel_size=4, padding='same', activation='tanh'
            )(final_output)
            final_output = Conv1D(
                filters=1, kernel_size=4, padding='same', activation='linear'
            )(final_output)
            final_output = Reshape(
                target_shape=(profile_length,), name=f"target_{target_profile_names[i]}"
            )(final_output)

            prof_act.append(final_output)

        model = Model(
            inputs=profile_inputs + actuator_past_inputs + actuator_future_inputs,
            outputs=prof_act,
        )

        name = 'test___ProfilePredictorConv1D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    def test_ProfilePredictorConv2D(self):
        input_profile_names = ['a', 'b', 'c']
        target_profile_names = ['a', 'b']
        actuator_names = ['aa', 'bb', 'cc']
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
                Input(shape=profile_inshape, name='input_' + input_profile_names[i])
            )
            profiles.append(
                Reshape((profile_lookback, profile_length, 1))(profile_inputs[i])
            )
        profiles = Concatenate(axis=-1)(profiles)
        # shape = (lookback, length, channels=num_profiles)
        profiles = Conv2D(
            filters=num_profiles * max_channels // 8,
            kernel_size=(1, profile_length // 12),
            strides=(1, 1),
            padding='same',
            activation=std_activation,
        )(profiles)
        profiles = Conv2D(
            filters=num_profiles * max_channels // 4,
            kernel_size=(1, profile_length // 8),
            strides=(1, 1),
            padding='same',
            activation=std_activation,
        )(profiles)
        profiles = Conv2D(
            filters=num_profiles * max_channels // 2,
            kernel_size=(1, profile_length // 6),
            strides=(1, 1),
            padding='same',
            activation=std_activation,
        )(profiles)
        profiles = Conv2D(
            filters=num_profiles * max_channels,
            kernel_size=(1, profile_length // 4),
            strides=(1, 1),
            padding='same',
            activation=std_activation,
        )(profiles)
        # shape = (lookback, length, channels)
        if profile_lookback > 1:
            profiles = Conv2D(
                filters=num_profiles * max_channels,
                kernel_size=(profile_lookback, 1),
                strides=(1, 1),
                padding='valid',
                activation=std_activation,
            )(profiles)
        profiles = Reshape(
            (profile_length, num_profiles * max_channels)
        )(profiles)
        # shape = (length, channels)

        actuator_future_inputs = []
        actuator_past_inputs = []
        actuators = []
        for i in range(num_actuators):
            actuator_future_inputs.append(
                Input(shape=future_actuator_inshape, name='input_future_' + actuator_names[i])
            )
            actuator_past_inputs.append(
                Input(shape=past_actuator_inshape, name='input_past_' + actuator_names[i])
            )
            actuators.append(
                Concatenate(axis=-1)([actuator_past_inputs[i], actuator_future_inputs[i]])
            )
            actuators[i] = Reshape((actuator_lookback + lookahead, 1))(actuators[i])
        actuators = Concatenate(axis=-1)(actuators)
        # shape = (time, num_actuators)
        actuators = Dense(
            units=num_profiles * max_channels // 8, activation=std_activation
        )(actuators)
        actuators = Dense(
            units=num_profiles * max_channels // 4, activation=std_activation
        )(actuators)
        actuators = Dense(
            units=num_profiles * max_channels // 2, activation=std_activation
        )(actuators)
        # Update recurrent_activation to 'sigmoid' as 'hard_sigmoid' is deprecated
        actuators = LSTM(
            units=num_profiles * max_channels,
            activation=std_activation,
            recurrent_activation='sigmoid',
        )(actuators)
        actuators = Reshape((num_profiles * max_channels, 1))(actuators)
        # shape = (channels, 1)
        actuators = Dense(
            units=profile_length // 4, activation=std_activation
        )(actuators)
        actuators = Dense(
            units=profile_length // 2, activation=std_activation
        )(actuators)
        actuators = Dense(units=profile_length, activation=None)(actuators)
        # shape = (channels, profile_length)
        actuators = Permute(dims=(2, 1))(actuators)
        # shape = (profile_length, channels)

        merged = Add()([profiles, actuators])
        merged = Reshape(
            (1, profile_length, num_profiles * max_channels)
        )(merged)
        # shape = (1, length, channels)

        prof_act = []
        for i in range(num_targets):
            prof_act.append(
                Conv2D(
                    filters=max_channels,
                    kernel_size=(1, profile_length // 4),
                    strides=(1, 1),
                    padding='same',
                    activation=std_activation,
                )(merged)
            )
            # shape = (1, length, max_channels)
            prof_act[i] = Conv2D(
                filters=max_channels // 2,
                kernel_size=(1, profile_length // 8),
                strides=(1, 1),
                padding='same',
                activation=std_activation,
            )(prof_act[i])
            prof_act[i] = Conv2D(
                filters=max_channels // 4,
                kernel_size=(1, profile_length // 6),
                strides=(1, 1),
                padding='same',
                activation=std_activation,
            )(prof_act[i])
            prof_act[i] = Conv2D(
                filters=max_channels // 8,
                kernel_size=(1, profile_length // 4),
                strides=(1, 1),
                padding='same',
                activation=std_activation,
            )(prof_act[i])
            prof_act[i] = Conv2D(
                filters=1,
                kernel_size=(1, profile_length // 4),
                strides=(1, 1),
                padding='same',
                activation=None,
            )(prof_act[i])
            # shape = (1, length, 1)
            prof_act[i] = Reshape(
                (profile_length,), name='target_' + target_profile_names[i]
            )(prof_act[i])
        model = Model(
            inputs=profile_inputs + actuator_past_inputs + actuator_future_inputs,
            outputs=prof_act,
        )
        name = 'test___ProfilePredictorConv2D' + str(int(time.time()))
        keras2c_main.k2c(model, name)
        rcode = build_and_run(name)
        self.assertEqual(rcode, 0)

    # The following test is commented out as it may require additional updates
    # def test_BabyMemNN(self):
    #     # Implementation of test_BabyMemNN
    #     pass  # Placeholder for future updates
