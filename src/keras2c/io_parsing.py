"""io_parsing.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Helper functions to get input and output names for each layer etc.
"""
from .backend import keras

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


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


def is_multi_input(layer):
    # If `layer.input` is a list, it's a multi-input layer
    return isinstance(layer.input, list)


def get_layer_num_io(layer):
    """Gets the number of inputs and outputs for a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        num_inputs (int): number of input nodes to the layer
        num_outputs (int): number of output nodes from the layer
    """
    # Initialize input and output counts
    num_inputs = 0
    num_outputs = 0

    # Handle InputLayer separately
    if isinstance(layer, keras.layers.InputLayer):
        num_inputs = 0  # InputLayer has no inbound nodes
        num_outputs = 1  # It produces one output tensor
    else:
        # Number of inputs
        if hasattr(layer, 'inputs'):
            inputs = layer.inputs
        elif hasattr(layer, 'input'):
            inputs = layer.input
        else:
            inputs = None

        if inputs is not None:
            if isinstance(inputs, list):
                num_inputs = len(inputs)
            else:
                num_inputs = 1

        # Number of outputs
        if hasattr(layer, 'outputs'):
            outputs = layer.outputs
        elif hasattr(layer, 'output'):
            outputs = layer.output
        else:
            outputs = None

        if outputs is not None:
            if isinstance(outputs, list):
                num_outputs = len(outputs)
            else:
                num_outputs = 1



    return num_inputs, num_outputs


def get_layer_io_names(layer):
    """Gets the names of the inputs and outputs of a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        inputs (list): names of all the input nodes to the layer
        outputs (list): names of all the output nodes from the layer
    """
    inputs = []
    outputs = []

    # Handle InputLayer separately
    if isinstance(layer, keras.layers.InputLayer):
        # InputLayer has no inputs, only an output
        name = layer.output.name.split(':')[0].split('/')[0]
        outputs.append(name)
    else:
        # Handle layers with multiple inputs
        if hasattr(layer, 'input'):
            input_tensors = layer.input
            if isinstance(input_tensors, list):
                for tensor in input_tensors:
                    name = tensor.name.split(':')[0].split('/')[0]
                    inputs.append(name)
            else:
                name = input_tensors.name.split(':')[0].split('/')[0]
                inputs.append(name)
        elif hasattr(layer, 'inputs'):
            input_tensors = layer.inputs
            if isinstance(input_tensors, list):
                for tensor in input_tensors:
                    name = tensor.name.split(':')[0].split('/')[0]
                    inputs.append(name)
            else:
                name = input_tensors.name.split(':')[0].split('/')[0]
                inputs.append(name)
        else:
            pass

        # Handle layers with multiple outputs
        if hasattr(layer, 'output'):
            output_tensors = layer.output
            if isinstance(output_tensors, list):
                for tensor in output_tensors:
                    name = tensor.name.split(':')[0].split('/')[0]
                    outputs.append(name)
            else:
                name = output_tensors.name.split(':')[0].split('/')[0]
                outputs.append(name)
        elif hasattr(layer, 'outputs'):
            output_tensors = layer.outputs
            if isinstance(output_tensors, list):
                for tensor in output_tensors:
                    name = tensor.name.split(':')[0].split('/')[0]
                    outputs.append(name)
            else:
                name = output_tensors.name.split(':')[0].split('/')[0]
                outputs.append(name)
        else:
            pass



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
        nm = model.inputs[i].name.split(':')[0].split('/')[0]
        inputs.append(nm)
    for i in range(num_outputs):
        nm = model.outputs[i].name.split(':')[0].split('/')[0]
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
