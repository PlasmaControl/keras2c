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


def get_model_layers(model):
    """Gets all layers/operations in the model that need code generation.

    In Keras 3, some operations (like Split) appear in model._operations
    but not in model.layers. This function returns a combined list.

    Args:
        model (keras Model): model to parse

    Returns:
        layers (list): list of all layers/operations
    """
    layers = list(model.layers)
    seen_names = {l.name for l in layers}
    if hasattr(model, '_operations'):
        for op in model._operations:
            if op.name not in seen_names:
                layers.append(op)
                seen_names.add(op.name)
    return layers


def get_real_tensor_names(model):
    """Gets the set of tensor names that are part of the real model graph.

    Traces backward from model outputs through inbound nodes, collecting all
    tensor names that are reachable. This filters out internal sub-layer
    tensors (e.g., from Bidirectional's internal forward/backward calls).

    Args:
        model (keras Model): model to parse

    Returns:
        real_names (set): set of tensor names in the real model graph
    """
    visited = set()
    queue = []
    for t in model.outputs:
        queue.append(t)
    for t in model.inputs:
        queue.append(t)

    all_layers = get_model_layers(model)

    while queue:
        t = queue.pop(0)
        tname = parse_io_name(t.name)
        if tname in visited:
            continue
        visited.add(tname)
        for layer in all_layers:
            for node in getattr(layer, '_inbound_nodes', []):
                out_t = getattr(node, 'output_tensors', None)
                if out_t is None:
                    continue
                if not isinstance(out_t, (list, tuple)):
                    out_t = [out_t]
                matched = False
                for ot in out_t:
                    if parse_io_name(ot.name) == tname:
                        matched = True
                        break
                if matched:
                    inp_t = node.input_tensors
                    if inp_t is not None:
                        if not isinstance(inp_t, (list, tuple)):
                            inp_t = [inp_t]
                        for it in inp_t:
                            queue.append(it)
    return visited


def get_all_io_names(model):
    """Gets names of all  node names in the model

    Args:
        model (keras Model): model to parse

    Returns:
        io (list): names of all the nodes in the model
    """

    valid = get_real_tensor_names(model)
    a = [get_layer_io_names(layer, valid) for layer in get_model_layers(model)]
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


def get_layer_io_names(layer, valid_tensors=None):
    """Gets the names of the inputs and outputs of a layer

    Args:
        layer (keras Layer): layer you want to parse
        valid_tensors (set, optional): if provided, only include nodes whose
            output tensors are in this set. Used to filter out internal
            sub-layer nodes (e.g., from Bidirectional wrappers).

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
            node_inp = []
        else:
            if isinstance(node_inputs, (list, tuple)):
                if len(node_inputs) == 1:
                    node_inp = parse_io_name(node_inputs[0].name)
                else:
                    node_inp = [parse_io_name(t.name) for t in node_inputs]
            else:
                # single tensor
                node_inp = parse_io_name(node_inputs.name)

        node_outputs = getattr(node, "output_tensors", None)
        if node_outputs is None:
            node_out = []
        else:
            if isinstance(node_outputs, (list, tuple)):
                if len(node_outputs) == 1:
                    node_out = parse_io_name(node_outputs[0].name)
                else:
                    node_out = [parse_io_name(t.name) for t in node_outputs]
            else:
                node_out = parse_io_name(node_outputs.name)

        # Filter: if valid_tensors provided, only include nodes whose outputs
        # are in the valid set (filters out internal sub-layer nodes)
        if valid_tensors is not None:
            flat_out = flatten([node_out]) if node_out else []
            if not any(o in valid_tensors for o in flat_out):
                continue

        inputs.append(node_inp)
        outputs.append(node_out)

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
