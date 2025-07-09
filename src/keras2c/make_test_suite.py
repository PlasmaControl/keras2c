"""make_test_suite.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Generates automatic test suite for converted code
"""

# Imports
import numpy as np
from keras2c.io_parsing import get_model_io_names
from keras2c.weights2c import Weights2C
import subprocess

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def make_test_suite(
    model,
    function_name,
    malloc_vars,
    num_tests=10,
    stateful=False,
    verbose=True,
    tol=1e-5,
):
    """Generates code to test the generated C function.

    Generates random inputs to the model and gets the corresponding predictions for them.
    Writes input/output pairs to a C file, along with code to call the generated C function
    and compare the true outputs with the outputs from the generated code.

    Writes the test function to a file `<function_name>_test_suite.c`

    Args:
        model (keras.Model): model being converted to C
        function_name (str): name of the neural net function being generated
        malloc_vars (dict): dictionary of names and values of variables allocated on the heap
        num_tests (int): number of tests to generate
        stateful (bool): whether the model contains layers that maintain state between calls
        verbose (bool): whether to print output
        tol (float): tolerance for passing tests. Tests pass if the maximum error over
            all elements between the true output and generated code output is less than tol

    Returns:
        None
    """

    if verbose:
        print("Writing tests")
    input_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)

    for i in range(num_inputs):
        temp_input_shape = model.inputs[i].shape
        temp_input_shape = [1 if dim is None else dim for dim in temp_input_shape]
        if stateful:
            temp_input_shape = temp_input_shape[:]
        else:
            temp_input_shape = temp_input_shape[1:]  # Exclude batch dimension
        input_shape.append(temp_input_shape)

    file = open(function_name + "_test_suite.c", "w")
    s = '#include <stdio.h> \n'
    s += '#include <math.h> \n'
    s += '#include <time.h> \n'
    s += '#include "./include/k2c_include.h" \n'
    s += f'#include "{function_name}.h" \n\n'
    s += "float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);\n"
    s += "struct timeval GetTimeStamp(); \n \n"
    file.write(s)

    for i in range(num_tests):
        if i == num_tests // 2 and stateful:
            # Reset states of stateful layers
            for layer in model.layers:
                if hasattr(layer, 'reset_states') and callable(
                        layer.reset_states):
                    layer.reset_states()
        # Generate random input and write to file
        ct = 0
        while True:
            rand_inputs = []
            for j in range(num_inputs):
                inp_layer = model.inputs[j]
                dt = getattr(inp_layer, 'dtype', None)
                if dt is not None and hasattr(dt, 'as_numpy_dtype'):
                    dt = dt.as_numpy_dtype
                if dt is not None and np.issubdtype(np.dtype(dt), np.integer):
                    high = 10
                    # attempt to infer size from connected Embedding layer
                    for layer in model.layers:
                        for node in layer._inbound_nodes:
                            inputs = getattr(node, 'input_tensors', [])
                            if not isinstance(inputs, (list, tuple)):
                                inputs = [inputs]
                            if inp_layer in inputs:
                                if hasattr(layer, 'input_dim'):
                                    high = layer.input_dim
                                break
                    rand_input = np.random.randint(0, high, size=tuple(input_shape[j]), dtype=np.dtype(dt))
                else:
                    rand_input = 4 * np.random.random(size=tuple(input_shape[j])) - 2
                if not stateful:
                    rand_input = rand_input[np.newaxis, ...]
                rand_inputs.append(rand_input)
            # Make predictions
            pred_input = rand_inputs if num_inputs > 1 else rand_inputs[0]
            outputs = model.predict(pred_input)
            if isinstance(outputs, list):
                outputs_concat = np.concatenate([np.ravel(o) for o in outputs])
            else:
                outputs_concat = outputs
            if np.isfinite(outputs_concat).all():
                break
            else:
                ct += 1
            if ct > 20:
                raise Exception(
                    "Cannot find inputs to the network that result in a finite output"
                )
        for j in range(num_inputs):
            file.write(
                Weights2C.array2c(
                    rand_inputs[j][0, :],
                    f"test{i + 1}_{model_inputs[j]}_input",
                )
            )

        # Write predictions
        if not isinstance(outputs, list):
            outputs = [outputs]
        for j in range(num_outputs):
            output = outputs[j][0, :]
            file.write(
                Weights2C.array2c(
                    output,
                    f"keras_{model_outputs[j]}_test{i + 1}",
                )
            )
            file.write(
                Weights2C.array2c(
                    np.zeros(output.shape),
                    f"c_{model_outputs[j]}_test{i + 1}",
                )
            )

    s = "int main(){\n"
    file.write(s)

    s = f" float errors[{num_tests * num_outputs}];\n"
    s += f" size_t num_tests = {num_tests}; \n"
    s += f"size_t num_outputs = {num_outputs}; \n"
    for var in malloc_vars:
        s += f"float* {var}; \n"

    init_sig = (
        f"{function_name}_initialize("
        + ",".join([f"&{var}" for var in malloc_vars])
        + "); \n"
    )
    s += init_sig
    if stateful:
        reset_sig = f"{function_name}_reset_states();\n"
        s += reset_sig
    s += "clock_t t0 = clock(); \n"
    file.write(s)

    for i in range(num_tests):
        if i == num_tests // 2 and stateful:
            file.write(reset_sig)
        s = f"{function_name}("
        model_in = [
            f"&test{i + 1}_{inp}_input" for inp in model_inputs
        ]
        model_out = [
            f"&c_{outp}_test{i + 1}" for outp in model_outputs
        ]
        s += ",".join(model_in + model_out + list(malloc_vars))
        s += "); \n"
        file.write(s)
    file.write("\n")
    s = "clock_t t1 = clock(); \n"
    s += (
        f'printf("Average time over {num_tests} tests: %e s \\n",\n'
        f'        ((double)t1-t0)/(double)CLOCKS_PER_SEC/(double){num_tests}); \n'
    )
    file.write(s)

    for i in range(num_tests):
        for j in range(num_outputs):
            s = (
                f"errors[{i * num_outputs + j}] = "
                f"maxabs(&keras_{model_outputs[j]}_test{i + 1},"
                f"&c_{model_outputs[j]}_test{i + 1}); \n"
            )
            file.write(s)
    s = "float maxerror = errors[0]; \n"
    s += "for(size_t i=1; i< num_tests*num_outputs;i++){ \n"
    s += "if (errors[i] > maxerror) { \n"
    s += "maxerror = errors[i];}} \n"
    s += f'printf("Max absolute error for {num_tests} tests: %e \\n", maxerror);\n'
    file.write(s)

    s = f"{function_name}_terminate(" + ",".join(malloc_vars) + "); \n"
    s += f"if (maxerror > {tol}) {{ \n"
    s += "return 1;} \n"
    s += "return 0;\n} \n\n"
    file.write(s)
    s = """float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){ \n
    float x = 0; \n
    float y = 0; \n
    for(size_t i=0; i<tensor1->numel; i++){\n
    y = fabsf(tensor1->array[i]-tensor2->array[i]);
    if (y>x) {x=y;}}
    return x;}\n\n"""
    file.write(s)
    file.close()
    try:
        subprocess.run(["astyle", "-n", function_name + "_test_suite.c"])
    except FileNotFoundError:
        print(
            f"astyle not found, {function_name}_test_suite.c will not be auto-formatted"
        )
