"""io_parsing.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3 License
https://github.com/f0uriest/keras2c

Helper functions to get input and output names for each layer etc.
"""

# Original author
# __author__ = "Rory Conlin"
# __copyright__ = "Copyright 2020, Rory Conlin"
# __license__ = "MIT"
# __maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
# __email__ = "wconlin@princeton.edu"

# Modified by
__author__ = "Anchal Gupta"
__email__ = "guptaa@fusion.gat.com"

def layer_type(layer):
    """Gets the type of a layer

    Args:
        layer (keras Layer): layer you want the type of

    Returns:
        type (str): what kind of layer it is. Eg "Dense", "Conv2D", "SimpleRNN"
    """

    return layer.__class__.__name__


def get_all_io_names(model):
    """Gets names of all  node names in the model

    Args:
        model (keras Model): model to parse

    Returns:
        io (list): names of all the nodes in the model
    """

    a = [get_layer_io_names(layer) for layer in model.layers]
    return list(set(flatten(a)))

def parse_io_name(name):
    name = name.replace('.', '_')
    skip_start = name.find('/')
    skip_end = name.rfind(':')
    out_str = name
    if skip_start != -1:
        out_str = name[:skip_start]
    if skip_end != -1:
        out_str += '_' + name[skip_end+1:]
    out_str = out_str.replace(':', '_').replace('/', '_')
    return out_str

def get_layer_num_io(layer):
    """Gets the number of inputs and outputs for a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        num_inputs (int): number of input nodes to the layer
        num_outputs (int): number of output nodes from the layer
    """

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
    """Gets the names of the inputs and outputs of a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        inputs (list): names of all the input nodes to the layer
        outputs (list): names of all the output nodes from the layer
    """

    num_inputs, num_outputs = get_layer_num_io(layer)
    inputs = []
    # num_inputs>1 -> shared layer
    for i in range(num_inputs):
        # is the input a list?
        if isinstance(layer.get_input_at(i), list):
            temp_list = []
            list_length = len(layer.get_input_at(i))
            for j in range(list_length):
                name = parse_io_name(layer.get_input_at(i)[j].name)
                # name = layer.get_input_at(i)[j].name.replace(':', '_').replace('/', '_').replace('.', '_')
                temp_list.append(name)
            inputs.insert(i, temp_list)
        else:
            name = parse_io_name(layer.get_input_at(i).name)
            # name = layer.get_input_at(i).name.replace(':', '_').replace('/', '_').replace('.', '_')
            inputs.insert(i, name)

    outputs = []
    for i in range(num_outputs):
        # is the output a list?
        if isinstance(layer.get_output_at(i), list):
            temp_list = []
            list_length = len(layer.get_output_at(i))
            for j in range(list_length):
                name = parse_io_name(layer.get_output_at(i)[j].name)
                # name = layer.get_output_at(i)[j].name.replace(':', '_').replace('/', '_').replace('.', '_')
                temp_list.append(name)
            outputs.insert(i, temp_list)
        else:
            name = layer.get_output_at(i).name
            name = parse_io_name(name)
            # if 'bidirectional' in name.lower():
            #     name = name.replace(':', '_').replace('/', '_').replace('.', '_')
            # else:
            #     name = name.replace(':', '_').replace('/', '_').replace('.', '_')
            outputs.insert(i, name)

    return inputs, outputs


def get_model_io_names(model):
    """Gets names of the input and output nodes of the model

    Args:
        model (keras Model): model to parse

    Returns:
        inputs (list): names of all the input nodes
        outputs (list): names of all the output nodes
    """

    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)
    inputs = []
    outputs = []
    for i in range(num_inputs):
        nm = parse_io_name(model.inputs[i].name)
        inputs.append(nm)
    for i in range(num_outputs):
        nm = parse_io_name(model.outputs[i].name)
        outputs.append(nm)
    return inputs, outputs


def flatten(x):
    """Flattens a nested list or tuple

    Args:
        x (list or tuple): nested list or tuple of lists or tuples to flatten

    Returns:
        x (list): flattened input
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
