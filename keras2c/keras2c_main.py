"""keras2c_main.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3 License
https://github.com/f0uriest/keras2c

Converts keras model to C code
"""

# imports
from keras2c.layer2c import Layers2C
from keras2c.weights2c import Weights2C
from keras2c.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten, get_model_layers, get_real_tensor_names
from keras2c.check_model import check_model
from keras2c.make_test_suite import make_test_suite
import numpy as np
import subprocess
import keras


# Original author
# __author__ = "Rory Conlin"
# __copyright__ = "Copyright 2020, Rory Conlin"
# __license__ = "MIT"
# __maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
# __email__ = "wconlin@princeton.edu"

# Modified by
__author__ = "Anchal Gupta"
__email__ = "guptaa@fusion.gat.com"


def fold_batch_norms(model, verbose=True):
    """Folds BatchNormalization layers into following Dense/Conv1D layers.
    Modifies model weights in-place. Returns set of folded layer names."""
    valid = get_real_tensor_names(model)
    consumers = {}
    for layer in get_model_layers(model):
        inputs, _ = get_layer_io_names(layer, valid)
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, list):
                    for ii in inp:
                        consumers.setdefault(ii, []).append(layer)
                elif isinstance(inp, str):
                    consumers.setdefault(inp, []).append(layer)
    folded_layers = set()
    for bn_layer in get_model_layers(model):
        if layer_type(bn_layer) != 'BatchNormalization':
            continue
        _, outputs = get_layer_io_names(bn_layer, valid)
        out_name = outputs[0] if isinstance(outputs, list) else outputs
        if isinstance(out_name, list):
            out_name = out_name[0]
        next_layers = consumers.get(out_name, [])
        if len(next_layers) != 1:
            continue
        next_layer = next_layers[0]
        next_type = layer_type(next_layer)
        if next_type not in ('Dense', 'Conv1D'):
            continue
        cfg = bn_layer.get_config()
        center = cfg['center']
        scale = cfg['scale']
        epsilon = cfg['epsilon']
        if center and scale:
            gamma, beta, mean, variance = bn_layer.get_weights()
        elif center:
            beta, mean, variance = bn_layer.get_weights()
            gamma = np.ones_like(mean)
        elif scale:
            gamma, mean, variance = bn_layer.get_weights()
            beta = np.zeros_like(mean)
        else:
            mean, variance = bn_layer.get_weights()
            gamma = np.ones_like(mean)
            beta = np.zeros_like(mean)
        stdev = np.sqrt(variance + epsilon)
        bn_scale = gamma / stdev
        bn_offset = beta - mean * bn_scale
        if next_type == 'Dense':
            weights = next_layer.get_weights()
            W = weights[0]
            b = weights[1] if len(weights) > 1 else np.zeros(W.shape[1])
            W_new = W * bn_scale[:, np.newaxis]
            b_new = bn_offset @ W + b
            if len(weights) > 1:
                next_layer.set_weights([W_new, b_new])
            else:
                if np.any(np.abs(b_new) > 1e-10):
                    continue
                next_layer.set_weights([W_new])
        elif next_type == 'Conv1D':
            weights = next_layer.get_weights()
            K = weights[0]
            b = weights[1] if len(weights) > 1 else np.zeros(K.shape[-1])
            K_new = K * bn_scale[np.newaxis, :, np.newaxis]
            b_new = np.tensordot(bn_offset, K, axes=([0], [1])).sum(axis=0) + b
            if len(weights) > 1:
                next_layer.set_weights([K_new, b_new])
            else:
                if np.any(np.abs(b_new) > 1e-10):
                    continue
                next_layer.set_weights([K_new])
        identity_var = np.ones_like(variance) * (1.0 - epsilon)
        if center and scale:
            bn_layer.set_weights([np.ones_like(gamma), np.zeros_like(beta),
                                  np.zeros_like(mean), identity_var])
        elif center:
            bn_layer.set_weights([np.zeros_like(beta), np.zeros_like(mean), identity_var])
        elif scale:
            bn_layer.set_weights([np.ones_like(gamma), np.zeros_like(mean), identity_var])
        else:
            bn_layer.set_weights([np.zeros_like(mean), identity_var])
        folded_layers.add(bn_layer.name)
        if verbose:
            print(f'Folded {bn_layer.name} into {next_layer.name}')
    return folded_layers


def model2c(model, function_name, malloc=False, verbose=True, skip_layers=None):
    """Generates C code for model

    Writes main function definition to "function_name.c" and a public header
    with declarations to "function_name.h"

    Args:
        model (keras Model): model to convert
        function_name (str): name of C function
        malloc (bool): whether to allocate variables on the stack or heap
        verbose (bool): whether to print info to stdout

    Returns:
        malloc_vars (list): names of variables loaded at runtime and stored on the heap
        stateful (bool): whether the model must maintain state between calls
    """

    if skip_layers is None:
        skip_layers = set()

    model_inputs, model_outputs = get_model_io_names(model)
    includes = '#include <math.h> \n '
    includes += '#include <string.h> \n'
    includes += '#include "./include/k2c_include.h" \n'
    includes += '#include "./include/k2c_tensor_include.h" \n'
    includes += '\n \n'

    if verbose:
        print('Gathering Weights')
    stack_vars, malloc_vars, static_vars = Weights2C(
        model, function_name, malloc).write_weights(verbose, skip_layers)
    stateful = len(static_vars) > 0
    layers = Layers2C(model, malloc).write_layers(verbose, skip_layers)

    function_signature = 'void ' + function_name + '('
    function_signature += ', '.join(['k2c_tensor* ' +
                                     in_nm + '_input' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['k2c_tensor* ' +
                                     out_nm + '_output' for out_nm in model_outputs])
    if len(malloc_vars.keys()):
        function_signature += ',' + ','.join(['float* ' +
                                              key for key in malloc_vars.keys()])
    function_signature += ')'

    init_sig, init_fun = gen_function_initialize(function_name, malloc_vars)
    term_sig, term_fun = gen_function_terminate(function_name, malloc_vars)
    reset_sig, reset_fun = gen_function_reset(function_name)

    with open(function_name + '.c', 'x+') as source:
        source.write(includes)
        source.write(static_vars + '\n\n')
        source.write(function_signature)
        source.write(' { \n\n')
        source.write(stack_vars)
        source.write(layers)
        source.write('\n } \n\n')
        source.write(init_fun)
        source.write(term_fun)
        if stateful:
            source.write(reset_fun)

    with open(function_name + '.h', 'x+') as header:
        header.write('#pragma once \n')
        header.write('#include "./include/k2c_tensor_include.h" \n')
        header.write(function_signature + '; \n')
        header.write(init_sig + '; \n')
        header.write(term_sig + '; \n')
        if stateful:
            header.write(reset_sig + '; \n')
    try:
        subprocess.run(['astyle', '-n', function_name + '.h'])
        subprocess.run(['astyle', '-n', function_name + '.c'])
    except FileNotFoundError:
        print("astyle not found, {} and {} will not be auto-formatted".format(function_name + ".h", function_name + ".c"))

    return malloc_vars.keys(), stateful


def gen_function_reset(function_name):
    """Writes a reset function for stateful models

    Reset function is used to clear internal state of the model

    Args:
        function_name (str): name of main function

    Returns:
       signature (str): delcaration of the reset function
       function (str): definition of the reset function
    """

    reset_sig = 'void ' + function_name + '_reset_states()'

    reset_fun = reset_sig
    reset_fun += ' { \n\n'
    reset_fun += 'memset(&' + function_name + \
                 '_states,0,sizeof(' + function_name + '_states)); \n'
    reset_fun += "} \n\n"
    return reset_sig, reset_fun


def gen_function_initialize(function_name, malloc_vars):
    """Writes an initialize function

    Initialize function is used to load variables into memory and do other start up tasks

    Args:
        function_name (str): name of main function
        malloc_vars (dict): variables to read in

    Returns:
       signature (str): delcaration of the initialization function
       function (str): definition of the initialization function
    """

    init_sig = 'void ' + function_name + '_initialize('
    init_sig += ','.join(['float** ' +
                          key + ' \n' for key in malloc_vars.keys()])
    init_sig += ')'

    init_fun = init_sig
    init_fun += ' { \n\n'
    for key in malloc_vars.keys():
        fname = function_name + key + ".csv"
        np.savetxt(fname, malloc_vars[key], fmt="%.8e", delimiter=',')
        init_fun += '*' + key + " = k2c_read_array(\"" + \
            fname + "\"," + str(malloc_vars[key].size) + "); \n"
    init_fun += "} \n\n"

    return init_sig, init_fun


def gen_function_terminate(function_name, malloc_vars):
    """Writes a terminate function

    Terminate function is used to deallocate memory after completion

    Args:
        function_name (str): name of main function
        malloc_vars (dict): variables to deallocate

    Returns:
       signature (str): delcaration of the terminate function
       function (str): definition of the terminate function
    """

    term_sig = 'void ' + function_name + '_terminate('
    term_sig += ','.join(['float* ' +
                          key for key in malloc_vars.keys()])
    term_sig += ')'

    term_fun = term_sig
    term_fun += ' { \n\n'
    for key in malloc_vars.keys():
        term_fun += "free(" + key + "); \n"
    term_fun += "} \n\n"

    return term_sig, term_fun


def k2c(model, function_name, malloc=False, num_tests=10, verbose=True):
    """Converts keras model to C code and generates test suite

    Args:
        model (keras Model or str): model to convert or path to saved .h5 file
        function_name (str): name of main function
        malloc (bool): whether to allocate variables on the stack or heap
        num_tests (int): how many tests to generate in the test suite
        verbose (bool): whether to print progress

    Raises:
        ValueError: if model is not instance of keras.models.Model

    Returns:
        None
    """

    function_name = str(function_name)
    filename = function_name + '.c'
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    elif not isinstance(model, keras.models.Model):

        raise ValueError('Unknown model type. Model should ' +
                         'either be an instance of keras.models.Model, ' +
                         'or a filepath to a saved .h5 model')

    # check that the model can be converted
    check_model(model, function_name)
    if verbose:
        print('All checks passed')

    folded = fold_batch_norms(model, verbose)
    if verbose and folded:
        print(f'Folded {len(folded)} batch normalization layers')

    malloc_vars, stateful = model2c(
        model, function_name, malloc, verbose, skip_layers=folded)

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
