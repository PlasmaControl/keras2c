"""keras2c_main.py
This file is part of keras2c
Converts keras model to C code
"""

# imports
import keras
import numpy as np
from keras2c.make_test_suite import make_test_suite
from keras2c.check_model import check_model
from keras2c.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten
from keras2c.weights2c import Weights2C
from keras2c.layer2c import Layers2C


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def model2c(model, file, function_name, malloc=False):
    model_inputs, model_outputs = get_model_io_names(model)
    s = '#include <stdio.h> \n#include <stddef.h> \n#include <math.h> \n#include <string.h> \n'
    s += '#include <stdarg.h> \n#include "./include/k2c_include.h" \n'
    s += '\n \n'
    file.write(s)

    print('Gathering Weights')
    stack_vars, malloc_vars = Weights2C(model, malloc).write_weights()
    layers = Layers2C(model, malloc).write_layers()

    function_signature = 'void ' + function_name + '('
    function_signature += ', '.join(['k2c_tensor* ' +
                                     in_nm + '_input' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['k2c_tensor* ' +
                                     out_nm + '_output' for out_nm in model_outputs])
    if len(malloc_vars.keys()):
        function_signature += ',' + ','.join(['float* ' +
                                              key for key in malloc_vars.keys()])
    function_signature += ')'

    file.write(function_signature)
    file.write(' { \n\n')
    file.write(stack_vars)
    file.write(layers)
    file.write('\n } \n\n')

    init_sig = write_function_initialize(file, function_name, malloc_vars)
    term_sig = write_function_terminate(file, function_name, malloc_vars)

    with open(function_name + '.h', 'x+') as header:
        header.write('#pragma once \n')
        header.write(function_signature + '; \n')
        header.write(init_sig + '; \n')
        header.write(term_sig + '; \n')

    return malloc_vars.keys()


def write_function_initialize(file, function_name, malloc_vars):
    function_init_signature = 'void ' + function_name + '_initialize('
    function_init_signature += ','.join(['float** ' +
                                         key for key in malloc_vars.keys()])
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


def k2c(model, function_name, malloc=False, num_tests=10):

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
    print('All checks passed')

    file = open(filename, "x+")
    malloc_vars = model2c(model, file, function_name, malloc)
    file.close()
    s = 'Done \n'
    s += "C code is in '" + function_name + ".h' \n"
    if num_tests > 0:
        make_test_suite(model, function_name, malloc_vars, num_tests)
        s += "Tests are in '" + function_name + "_test_suite.c' \n"
    if malloc:
        s += "Weight arrays are in .csv files of the form 'model_name_layer_name_array_type.csv' \n"
        s += "They should be placed in the directory from which the main program is run."
    print(s)
