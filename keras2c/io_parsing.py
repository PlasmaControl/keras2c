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

    if hasattr(layer, "inputs"):
        if isinstance(layer.inputs, list):
            num_inputs = len(layer.inputs)
        else:
            num_inputs = 1
    else:
        # fallback: count inbound nodes
        num_inputs = len(getattr(layer, "_inbound_nodes", []))

    # If outputs attribute is present, count actual tensor outputs
    if hasattr(layer, "outputs"):
        outs = layer.outputs
        if isinstance(outs, list):
            num_outputs = len(outs)
        else:
            num_outputs = 1
    else:
        # Fallback: count graph nodes that produce outputs
        num_outputs = len(getattr(layer, "_inbound_nodes", []))
    return num_inputs, num_outputs


def get_layer_io_names(layer):
    """Gets the names of the inputs and outputs of a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        inputs (list): names of all the input nodes to the layer
        outputs (list): names of all the output nodes from the layer
    """

    num_nodes = len(getattr(layer, "_inbound_nodes", []))

    inputs = []
    outputs = []

    for node_index in range(num_nodes):
        node = layer._inbound_nodes[node_index]
        # is the input a list?
        node_inputs = node.input_tensors
        if node_inputs is None:
            inputs.append([])
        else:
            if isinstance(node_inputs, (list, tuple)):
                if len(node_inputs) == 1:
                    inputs.append(parse_io_name(node_inputs[0].name))
                else:
                    inputs.append([parse_io_name(t.name) for t in node_inputs])
            else:
                # single tensor
                inputs.append(parse_io_name(node_inputs.name))

        node_outputs = getattr(node, "output_tensors", None)
        if node_outputs is None:
            outputs.append([])
        else:
            if isinstance(node_outputs, (list, tuple)):
                if len(node_outputs) == 1:
                    outputs.append(parse_io_name(node_outputs[0].name))
                else:
                    outputs.append([parse_io_name(t.name) for t in node_outputs])
            else:
                outputs.append(parse_io_name(node_outputs.name))

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
