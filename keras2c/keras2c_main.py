"""keras2c_main.py
This file is part of keras2c
Converts keras model to C code
"""

# imports
from keras2c.layer2c import Layers2C
from keras2c.weights2c import Weights2C
from keras2c.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten
from keras2c.check_model import check_model
from keras2c.make_test_suite import make_test_suite
import numpy as np
import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def model2c(model, file, function_name, malloc=False, verbose=True):
    """Generates C code for model

    Writes main function definition to "function_name.c" and a public header 
    with declarations to "function_name.h"

    Args:
        model (keras Model): model to convert
        file (open file instance): where to write main function
        function_name (str): name of C function
        malloc (bool): whether to allocate variables on the stack or heap
        verbose (bool): whether to print info to stdout

    Returns:
        malloc_vars (list): names of variables loaded at runtime and stored on the heap
        stateful (bool): whether the model must maintain state between calls
    """

    model_inputs, model_outputs = get_model_io_names(model)
    s = '#include <math.h> \n '
    s += '#include <string.h> \n'
    s += '#include "./include/k2c_include.h" \n'
    s += '#include "./include/k2c_tensor_include.h" \n'
    s += '\n \n'
    file.write(s)

    if verbose:
        print('Gathering Weights')
    stack_vars, malloc_vars, static_vars = Weights2C(
        model, function_name, malloc).write_weights(verbose)
    layers = Layers2C(model, malloc).write_layers(verbose)

    function_signature = 'void ' + function_name + '('
    function_signature += ', '.join(['k2c_tensor* ' +
                                     in_nm + '_input' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['k2c_tensor* ' +
                                     out_nm + '_output' for out_nm in model_outputs])
    if len(malloc_vars.keys()):
        function_signature += ',' + ','.join(['float* ' +
                                              key for key in malloc_vars.keys()])
    function_signature += ')'
    file.write(static_vars + '\n\n')
    file.write(function_signature)
    file.write(' { \n\n')
    file.write(stack_vars)
    file.write(layers)
    file.write('\n } \n\n')
    stateful = len(static_vars) > 0

    init_sig = write_function_initialize(file, function_name, malloc_vars)
    term_sig = write_function_terminate(file, function_name, malloc_vars)
    if stateful:
        reset_sig = write_function_reset(file, function_name)
    with open(function_name + '.h', 'x+') as header:
        header.write('#pragma once \n')
        header.write('#include "./include/k2c_tensor_include.h" \n')
        header.write(function_signature + '; \n')
        header.write(init_sig + '; \n')
        header.write(term_sig + '; \n')

        if stateful:
            header.write(reset_sig + '; \n')

    return malloc_vars.keys(), stateful


def write_function_reset(file, function_name):
    """Writes a reset function for stateful models

    Reset function is used to clear internal state of the model

    Args:
        file (open file instance): file to write to
        function_name (str): name of main function

    Returns:
       signature (str): delcaration of the reset function
    """

    function_reset_signature = 'void ' + function_name + '_reset_states()'
    file.write(function_reset_signature)
    s = ' { \n\n'
    s += 'memset(&' + function_name + \
         '_states,0,sizeof(' + function_name + '_states)); \n'
    s += "} \n\n"
    file.write(s)
    return function_reset_signature


def write_function_initialize(file, function_name, malloc_vars):
    """Writes an initialize function

    Initialize function is used to load variables into memory and do other start up tasks

    Args:
        file (open file instance): file to write to
        function_name (str): name of main function

    Returns:
       signature (str): delcaration of the initialization function
    """

    function_init_signature = 'void ' + function_name + '_initialize('
    function_init_signature += ','.join(['float** ' +
                                         key + ' \n' for key in malloc_vars.keys()])
    function_init_signature += ')'
    file.write(function_init_signature)
    s = ' { \n\n'
    for key in malloc_vars.keys():
        fname = function_name + key + ".csv"
        np.savetxt(fname, malloc_vars[key], fmt="%.8e", delimiter=',')
        s += '*' + key + " = k2c_read_array(\"" + \
            fname + "\"," + str(malloc_vars[key].size) + "); \n"
    s += "} \n\n"
    file.write(s)
    return function_init_signature


def write_function_terminate(file, function_name, malloc_vars):
    """Writes a terminate function

    Terminate function is used to deallocate memory after completion

    Args:
        file (open file instance): file to write to
        function_name (str): name of main function

    Returns:
       signature (str): delcaration of the terminate function
    """

    function_term_signature = 'void ' + function_name + '_terminate('
    function_term_signature += ','.join(['float* ' +
                                         key for key in malloc_vars.keys()])
    function_term_signature += ')'
    file.write(function_term_signature)
    s = ' { \n\n'
    for key in malloc_vars.keys():
        s += "free(" + key + "); \n"
    s += "} \n\n"
    file.write(s)
    return function_term_signature


def k2c(model, function_name, malloc=False, num_tests=10, verbose=True):
    """Converts keras model to C code and generates test suite

    Args:
        model (keras Model or str): model to convert or path to saved .h5 file
        function_name (str): name of main function
        malloc (bool): whether to allocate variables on the stack or heap
        num_tests (int): how many tests to generate in the test suite

    Raises:
        ValueError: if model is not instance of keras.models.Model 
            or keras.engine.training.Model

    Returns:
        None
    """

    function_name = str(function_name)
    filename = function_name + '.c'
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    elif not isinstance(model, (keras.models.Model,
                                keras.engine.training.Model)):

        raise ValueError('Unknown model type. Model should ' +
                         'either be an instance of keras.models.Model, ' +
                         'or a filepath to a saved .h5 model')

    # check that the model can be converted
    check_model(model, function_name)
    if verbose:
        print('All checks passed')

    file = open(filename, "x+")
    malloc_vars, stateful = model2c(
        model, file, function_name, malloc, verbose)
    file.close()
    s = 'Done \n'
    s += "C code is in '" + function_name + \
        ".c' with header file '" + function_name + ".h' \n"
    if num_tests > 0:
        make_test_suite(model, function_name, malloc_vars,
                        num_tests, stateful, verbose)
        s += "Tests are in '" + function_name + "_test_suite.c' \n"
    if malloc:
        s += "Weight arrays are in .csv files of the form 'model_name_layer_name_array_type.csv' \n"
        s += "They should be placed in the directory from which the main program is run."
    if verbose:
        print(s)
