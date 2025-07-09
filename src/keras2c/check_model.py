"""check_model.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Checks a model before conversion to flag unsupported features
"""

# Imports
import numpy as np
from keras2c.io_parsing import layer_type, flatten
from keras2c.weights2c import Weights2C
from keras2c.layer2c import Layers2C


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def is_valid_c_name(name):
    """Checks if a name is a valid name for a C variable or function.

    Args:
        name (str): name to check

    Returns:
        valid (bool): 'True' if the name is valid, 'False' otherwise
    """

    allowed_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
    allowed_starting_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    if not set(name).issubset(allowed_chars) or not \
       set(name[0]).issubset(allowed_starting_chars):
        return False
    return True


def name_check(model):
    """Checks if all layer names in a model are valid C names.

    Args:
       model (keras.Model): model to check

    Returns:
        valid (bool): 'True' if all names are valid, 'False' otherwise
        log (str): log of invalid names
    """

    valid = True
    log = ''
    for layer in model.layers:
        if not is_valid_c_name(layer.name):
            valid = False
            log += f"Layer name '{layer.name}' is not a valid C name.\n"
    return valid, log


def layers_supported_check(model):
    """Checks if all layers in the model are supported

    Args:
       model (keras.Model): model to check

    Returns:
        valid (bool): 'True' if all layers are supported, 'False' otherwise
        log (str): log of unsupported layers
    """

    def check_layer(layer):
        valid = True
        log = ''
        if hasattr(layer, 'layer'):
            flag, templog = check_layer(layer.layer)
            valid = valid and flag
            log += templog
        if not hasattr(Weights2C, f'_write_weights_{layer_type(layer)}') \
           or not hasattr(Layers2C, f'_write_layer_{layer_type(layer)}'):
            valid = False
            log += f"Layer type '{layer_type(layer)}' is not supported at this time.\n"
        return valid, log

    valid = True
    log = ''
    for layer in model.layers:
        flag, templog = check_layer(layer)
        valid = valid and flag
        log += templog
    return valid, log


def activation_supported_check(model):
    """Checks if all activation functions in the model are supported

    Args:
       model (keras.Model): model to check

    Returns:
        valid (bool): 'True' if all activations are supported, 'False' otherwise
        log (str): log of unsupported activation functions
    """

    supported_activations = [
        'linear', 'relu', 'softmax', 'softplus',
        'softsign', 'tanh', 'sigmoid',
        'hard_sigmoid', 'exponential', 'selu', 'elu', 'gelu', 'swish', 'silu'
    ]

    def check_layer(layer):
        valid = True
        log = ''
        if hasattr(layer, 'layer'):
            flag, templog = check_layer(layer.layer)
            valid = valid and flag
            log += templog
        config = layer.get_config()
        activation = config.get('activation')
        recurrent_activation = config.get('recurrent_activation')
        if activation not in supported_activations and activation is not None:
            valid = False
            log += (
                f"Activation type '{activation}' for layer '{layer.name}' "
                "is not supported at this time.\n"
            )
        if recurrent_activation not in supported_activations and recurrent_activation is not None:
            valid = False
            log += (
                f"Recurrent activation type '{recurrent_activation}' for "
                f"layer '{layer.name}' is not supported at this time.\n"
            )
        return valid, log

    valid = True
    log = ''
    for layer in model.layers:
        flag, templog = check_layer(layer)
        valid = valid and flag
        log += templog
    return valid, log

# Add check for masking if necessary

def config_supported_check(model):
    """Checks if all layer features in the model are supported

    Args:
       model (keras.Model): model to check

    Returns:
        valid (bool): 'True' if all features are supported, 'False' otherwise
        log (str): log of unsupported features
    """

    def check_layer(layer):
        valid = True
        log = ''
        if hasattr(layer, 'layer'):
            flag, templog = check_layer(layer.layer)
            valid = valid and flag
            log += templog
        config = layer.get_config()
        if config.get('merge_mode', 'foo') is None:
            valid = False
            log += (
                "Merge mode of 'None' for Bidirectional layers is not "
                "supported. Try using two separate RNNs instead.\n"
            )
        if config.get('data_format') not in ['channels_last', None]:
            valid = False
            log += (
                f"Data format '{config.get('data_format')}' for layer "
                f"'{layer.name}' is not supported at this time.\n"
            )
        if config.get('return_state'):
            valid = False
            log += (
                f"'return_state' option for layer '{layer.name}' is "
                "not supported at this time.\n"
            )
        if config.get('shared_axes'):
            valid = False
            log += (
                f"'shared_axes' option for layer '{layer.name}' is "
                "not supported at this time.\n"
            )
        if layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Average',
                                 'Maximum', 'Minimum']:
            inshps = [tensor.shape for tensor in layer.input]
            if isinstance(inshps, list):
                insize = []
                for inp in inshps:
                    # Exclude batch dimension and replace None with 1
                    shape = [dim if dim is not None else 1 for dim in inp[1:]]
                    insize.append(np.prod(shape))
                if len(set(insize)) > 1:
                    valid = False
                    log += (
                        "Broadcasting merge functions between tensors of "
                        f"different shapes for layer '{layer.name}' is not "
                        "currently supported.\n"
                    )
        if layer_type(layer) in ['BatchNormalization']:
            if isinstance(config.get('axis'), (list, tuple)) and len(flatten(config.get('axis'))) > 1:
                valid = False
                log += 'Batch normalization along multiple axes is not currently supported.\n'
        return valid, log

    valid = True
    log = ''
    for layer in model.layers:
        flag, templog = check_layer(layer)
        valid = valid and flag
        log += templog
    return valid, log


def check_model(model, function_name):
    """Checks if all names are valid and all features are supported

    Args:
        model (keras.Model): model to check
        function_name (str): name of the function being created

    Raises:
        AssertionError: If model contains invalid names or unsupported features
    """

    valid_fname = True
    log = 'The following errors were found:\n'
    if not is_valid_c_name(function_name):
        valid_fname = False
        log += f"Function name '{function_name}' is not a valid C name.\n"
    valid_lname, name_log = name_check(model)
    log += name_log
    valid_layer, layer_log = layers_supported_check(model)
    log += layer_log
    valid_activation, activation_log = activation_supported_check(model)
    log += activation_log
    valid_config, config_log = config_supported_check(model)
    log += config_log
    if not (valid_fname and valid_lname and valid_layer and
            valid_activation and valid_config):
        raise AssertionError(log)
