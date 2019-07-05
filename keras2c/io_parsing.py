"""io_parsing.py
This file is part of keras2c
Helper functions to get input and output names for each layer etc.
"""

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


# types, names, io

def layer_type(layer):
    return str(layer.__class__).split('.')[-1][0:-2]


def get_all_io_names(model):
    a = [get_layer_io_names(layer) for layer in model.layers]
    return list(set(flatten(a)))


def get_layer_num_io(layer):
    num_inputs = 0
    error = False
    while not error:
        try:
            layer.get_input_at(num_inputs)
            num_inputs += 1
        except ValueError:
            error = True

    num_outputs = 0
    error = False
    while not error:
        try:
            layer.get_output_at(num_outputs)
            num_outputs += 1
        except ValueError:
            error = True
    return num_inputs, num_outputs


def get_layer_io_names(layer):
    num_inputs, num_outputs = get_layer_num_io(layer)
    inputs = []
    # num_inputs>1 -> shared layer
    for i in range(num_inputs):
        # is the input a list?
        if isinstance(layer.get_input_at(i), list):
            temp_list = []
            list_length = len(layer.get_input_at(i))
            for j in range(list_length):
                name = str(layer.get_input_at(i)[j]).split()[
                    0].split('"')[1].split('/')[0].split(':')[0]
                temp_list.append(name)
            inputs.insert(i, temp_list)
        else:
            name = str(layer.get_input_at(i)).split()[0].split(
                '"')[1].split('/')[0].split(':')[0]
            inputs.insert(i, name)

    outputs = []
    for i in range(num_outputs):
        # is the output a list?
        if isinstance(layer.get_output_at(i), list):
            temp_list = []
            list_length = len(layer.get_output_at(i))
            for j in range(list_length):
                name = str(layer.get_output_at(i)[j]).split()[
                    0].split('"')[1].split('/')[0].split(':')[0]
                temp_list.append(name)
            outputs.insert(i, temp_list)
        else:
            name = str(layer.get_output_at(i)).split()[
                0].split('"')[1].split('/')[0].split(':')[0]
            outputs.insert(i, name)

    return inputs, outputs


def get_model_io_names(model):
    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)
    inputs = []
    outputs = []
    for i in range(num_inputs):
        nm = str(model.inputs[i]).split()[0].split(
            '"')[1].split('/')[0].split(':')[0]
        inputs.append(nm)
    for i in range(num_outputs):
        nm = str(model.outputs[i]).split()[0].split(
            '"')[1].split('/')[0].split(':')[0]
        outputs.append(nm)
    return inputs, outputs


def flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
