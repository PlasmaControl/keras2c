"""make_test_suite.py
This file is part of keras2c
Converts keras model to C code
"""

# imports
import numpy as np
from keras2c.io_parsing import get_model_io_names
from keras2c.weights2c import Weights2C

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def make_test_suite(model, function_name, malloc_vars, num_tests=10, tol=1e-5):
    print('Writing tests')
    input_shape = []
    # output_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)
    for i in range(num_inputs):
        input_shape.insert(i, model.inputs[i].shape[1:])
  #  for i in range(num_outputs):
  #      output_shape.insert(i, model.outputs[i].shape[1:])

    file = open(function_name + '_test_suite.c', "x+")
    s = '#include <stdio.h> \n#include <math.h> \n#include <sys/time.h> \n#include "' + \
        function_name + '.h" \n\n'
    s += 'float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);\n'
    s += 'struct timeval GetTimeStamp(); \n \n'
    file.write(s)
    s = 'int main(){\n'
    file.write(s)
    for i in range(num_tests):
        # generate random input and write to file
        ct = 0
        while True:
            rand_inputs = []
            for j, _ in enumerate(model_inputs):
                rand_input = 4*np.random.random(input_shape[j]) - 2
                rand_input = rand_input[np.newaxis, ...]
                rand_inputs.insert(j, rand_input)
                # make predictions
            outputs = model.predict(rand_inputs)
            if np.isfinite(outputs).all():
                break
            if ct > 20:
                raise Exception('Cannot find inputs to the \
                network that result in a finite output')
        for j, _ in enumerate(model_inputs):
            file.write(Weights2C.array2c((rand_inputs[j][0, :]), 'test' + str(i+1) +
                                         '_' + model_inputs[j] + '_input'))

            # write predictions
        if not isinstance(outputs, list):
            outputs = [outputs]
            for j, _ in enumerate(model_outputs):
                output = outputs[j][0, :]
                file.write(Weights2C.array2c(output, 'keras_' +
                                             model_outputs[j] + '_test' + str(i+1)))
                file.write(Weights2C.array2c(np.zeros(output.shape), 'c_' +
                                             model_outputs[j] + '_test' + str(i+1)))
    s = ' float errors[' + str(num_tests*num_outputs) + '];\n'
    s += ' size_t num_tests = ' + str(num_tests) + '; \n'
    s += 'size_t num_outputs = ' + str(num_outputs) + '; \n'
    for var in malloc_vars:
        s += 'float* ' + var + '; \n'
    s += function_name + \
        '_initialize(' + ','.join(['&' + var for var in malloc_vars]) + '); \n'
    s += ' struct timeval t1 = GetTimeStamp(); \n'
    file.write(s)

    for i in range(num_tests):
        s = function_name + '('
        model_in = ['&test' + str(i+1) + '_' + inp +
                    '_input' for inp in model_inputs]
        model_out = ['&c_' + outp + '_test' +
                     str(i+1) for outp in model_outputs]
        s += ','.join(model_in + model_out + list(malloc_vars))
        s += '); \n'
        file.write(s)
    file.write('\n')
    s = 'struct timeval t2 = GetTimeStamp(); \n'
    s += 'typedef unsigned long long u64; \n'
    s += 'u64 t1u = t1.tv_sec*1e6 + t1.tv_usec; \n'
    s += 'u64 t2u = t2.tv_sec*1e6 + t2.tv_usec; \n'
    s += 'printf("Average time over ' + str(num_tests) + \
        ' tests: %llu us \\n", (t2u-t1u)/' + str(num_tests) + '); \n'
    file.write(s)

    for i in range(num_tests):
        for j, _ in enumerate(model_outputs):
            s = 'errors[' + str(i*num_outputs+j) + '] = maxabs(&keras_' + model_outputs[j] + '_test' + \
                str(i+1) + ',&c_' + \
                model_outputs[j] + '_test' + str(i+1) + '); \n'
            file.write(s)
    s = 'float maxerror = errors[0]; \n'
    s += 'for(size_t i=1; i< num_tests*num_outputs;i++){ \n'
    s += 'if (errors[i] > maxerror) { \n'
    s += 'maxerror = errors[i];}} \n'
    s += 'printf("Max absolute error for ' + \
        str(num_tests) + ' tests: %e \\n", maxerror);\n'
    file.write(s)

    # s = 'for(size_t i=0; i< num_tests*num_outputs;i++){ \n'
    # s += 'printf(\"Error, test %d: %f \\n \",i,errors[i]);} \n'
    # file.write(s)

    s = function_name + '_terminate(' + ','.join(malloc_vars) + '); \n'
    s += 'if (maxerror > ' + str(tol) + ') { \n'
    s += 'return 1;} \n'
    s += 'return 0;\n} \n\n'
    file.write(s)
    s = """float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){ \n
    float x = 0; \n
    float y = 0; \n
    for(size_t i=0; i<tensor1->numel; i++){\n
    y = fabs(tensor1->array[i]-tensor2->array[i]);
    if (y>x) {x=y;}}
    return x;}\n\n"""
    file.write(s)
    s = """struct timeval GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv;}"""
    file.write(s)
    file.close()
