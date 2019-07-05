"""keras2c_main.py
This file is part of keras2c
Converts keras model to C code
"""

# imports
import keras
from keras2c.make_test_suite import make_test_suite
from keras2c.check_model import check_model
from keras2c.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten
from keras2c.weights2c import Weights2C
from keras2c.layer2c import layer2c


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def model2c(model, file, function_name):
    model_inputs, model_outputs = get_model_io_names(model)

    s = '#include <stdio.h> \n#include <stddef.h> \n#include <math.h> \n#include <string.h> \n'
    s += '#include <stdarg.h> \n#include "k2c_include.h" \n'
    s += '\n \n'
    s += 'void ' + function_name + '('
    s_in = ['k2c_tensor* ' + in_nm + '_input' for in_nm in model_inputs]
    s += ', '.join(s_in) + ', '
    s_out = ['k2c_tensor* ' + out_nm + '_output' for out_nm in model_outputs]
    s += ', '.join(s_out) + ') { \n \n'
    file.write(s)

    print('Writing Weights')
    weights = Weights2C(model).write_weights()
    file.write(weights)
    print('Weights written')

    written_io = set(model_inputs)
    unwritten_io = set(get_all_io_names(model)) - written_io

    while len(unwritten_io) > 0:
        for layer in model.layers:
            layer_inputs, layer_outputs = get_layer_io_names(layer)
            for i, (inp, outp) in enumerate(zip(layer_inputs, layer_outputs)):
                if (set(flatten(inp)).issubset(written_io) and
                        set(flatten(outp)).issubset(unwritten_io))or \
                        layer_type(layer) == 'InputLayer':
                    print('Writing layer ', outp)
                    is_model_input = False
                    is_model_output = False
                    if isinstance(inp, list):
                        inp_nm = []
                        for j in inp:
                            if j in model_inputs:
                                inp_nm.append(j + '_input')
                                is_model_input = True
                            else:
                                inp_nm.append('&' + j + '_output')
                    else:
                        if inp in model_inputs:
                            inp_nm = inp + '_input'
                            is_model_input = True
                        else:
                            inp_nm = '&' + inp + '_output'
                    if isinstance(outp, list):
                        outp_nm = []
                        for o in outp:
                            if o in model_outputs:
                                outp_nm.append(outp + '_output')
                                is_model_output = True
                            else:
                                outp_nm.append('&' + outp + '_output')
                    else:
                        if outp in model_outputs:
                            outp_nm = outp + '_output'
                            is_model_output = True
                        else:
                            outp_nm = '&' + outp + '_output'

                    layer2c(layer, file, inp_nm, outp_nm,
                            i, is_model_input, is_model_output)
                    written_io |= set(flatten(inp))
                    written_io |= set(flatten(outp))
                    unwritten_io -= set(flatten(inp))
                    unwritten_io -= set(flatten(outp))
    file.write('\n }')


# keras2c
def k2c(model, function_name, num_tests=10):

    function_name = str(function_name)
    filename = function_name + '.h'
    if isinstance(model, str):
        model = keras.models.load_model(str(model_filepath))
    elif not isinstance(model, (keras.models.Model,
                                keras.engine.training.Model)):

        raise ValueError('Unknown model type. Model should ' +
                         'either be an instance of keras.models.Model, ' +
                         'or a filepath to a saved .h5 model')

    # check that the model can be converted
    check_model(model, function_name)
    print('All checks passed')

    file = open(filename, "x+")
    model2c(model, file, function_name)
    file.close()
    make_test_suite(model, function_name, num_tests)
    print("Done \n C code is in '" + function_name +
          ".h' and tests are in '" + function_name + "_test_suite.c'")
