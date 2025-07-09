"""weights2c.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Gets weights and other parameters from each layer and writes to C file
"""

# Imports
import numpy as np
from keras2c.io_parsing import layer_type, get_layer_io_names, get_model_io_names
from .backend import keras

maxndim = 5

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class Weights2C:
    """Creates an object to extract and write weights and other model parameters

    Args:
        model (keras.Model): model to parse
        function_name (str): name of the function being generated
        malloc (bool): Whether to allocate variables on the heap using malloc.
    """

    def __init__(self, model, function_name, malloc=False):
        self.model = model
        self.function_name = function_name
        self.model_io = get_model_io_names(self.model)
        self.malloc = malloc
        self.stack_vars = ''
        self.malloc_vars = {}
        self.static_vars = {}

    @staticmethod
    def array2c(array, name, malloc=False):
        """Generates C code for a k2c_tensor array type.

        Args:
            array (array-like): Python array to write
            name (str): name for the C variable
            malloc (bool): whether to allocate on the heap

        Returns:
            str: generated code for the array when ``malloc`` is ``False``.
            (str, dict): generated code and dictionary of arrays to allocate
                when ``malloc`` is ``True``.
        """
        temp = array.flatten(order='C')
        size = array.size
        shp = array.shape
        ndim = len(shp)
        shp = np.concatenate((shp, np.ones(maxndim - ndim)))
        if malloc:
            to_malloc = {}
            s = (
                f'k2c_tensor {name} = {{ {name}_array, {ndim}, {size}, '
                f'{{ {np.array2string(shp.astype(int), separator=",")[1:-1]} }} }}; \n'
            )
            to_malloc.update({f'{name}_array': temp})
            return s, to_malloc
        else:
            count = 0
            s = f'float {name}_array[{size}] = '
            if np.max(np.abs(temp)) < 1e-16:
                s += '{0}; \n'
            else:
                s += '{\n'
                for i in range(size):
                    if temp[i] == np.inf:
                        s += "HUGE_VALF,"
                    elif temp[i] == -np.inf:
                        s += "-HUGE_VALF,"
                    else:
                        s += f"{temp[i]:+.8e}f,"
                    count += 1
                    if count % 5 == 0:
                        s += '\n'
                s += '}; \n'
            s += (
                f'k2c_tensor {name} = {{ &{name}_array[0], {ndim}, {size}, '
                f'{{ {np.array2string(shp.astype(int), separator=",")[1:-1]} }} }}; \n'
            )
            return s

    def _write_weights_array2c(self, array, name):
        temp = self.array2c(array, name, self.malloc)
        if self.malloc:
            self.stack_vars += temp[0]
            self.malloc_vars.update(temp[1])
        else:
            self.stack_vars += temp

    def _write_weights_layer(self, layer):
        method = getattr(self, '_write_weights_' + layer_type(layer))
        return method(layer)

    def write_weights(self, verbose=True):
        """Parses and generates code for model weights and other parameters

        Args:
            verbose (bool): whether to print progress

        Returns:
            (tuple): tuple containing

                - **stack_vars** (*str*): code for variables allocated on the stack
                - **malloc_vars** (*dict*): dictionary of name,value pairs for arrays to be
                    allocated on the heap
                - **static_vars** (*str*): code for a C struct containing static variables
                    (e.g., states of a stateful RNN)
        """
        for layer in self.model.layers:
            method = getattr(self, '_write_weights_' + layer_type(layer))
            method(layer)
        return self.stack_vars, self.malloc_vars, self._write_static_vars()

    def _write_static_vars(self):
        if len(self.static_vars) > 0:
            s = f'static struct {self.function_name}_static_vars \n'
            s += '{ \n'
            for k, v in self.static_vars.items():
                s += f'float {k}[{v}]; \n'
            s += f'}} {self.function_name}_states; \n'
        else:
            s = ''
        return s

    def _write_outputs(self, layer):
        _, outputs = get_layer_io_names(layer)
        if len(outputs) > 1:
            for i, outp in enumerate(outputs):
                outshp = layer.output_shape[i][1:]
                if outp not in self.model_io[1]:
                    self._write_weights_array2c(
                        np.zeros(outshp), f'{outp}_output')
        else:
            outshp = layer.output.shape[1:]
            if outputs[0] not in self.model_io[1]:
                self._write_weights_array2c(
                    np.zeros(outshp), f'{outputs[0]}_output')

    def _write_weights_Bidirectional(self, layer):
        inputs, outputs = get_layer_io_names(layer)

        # Get the input shape of the Bidirectional layer (excluding batch dimension)
        input_shape = layer.input.shape  # This includes batch dimension
        input_shape_no_batch = input_shape[1:]  # Exclude batch dimension

        # Get the forward and backward layers
        forward_layer = layer.forward_layer
        backward_layer = layer.backward_layer

        # Compute output shapes for forward and backward layers
        # Include batch dimension as None when computing output shape
        forward_output_shape = forward_layer.compute_output_shape(
            (None,) + input_shape_no_batch)
        backward_output_shape = backward_layer.compute_output_shape(
            (None,) + input_shape_no_batch)

        # Exclude batch dimension from output shapes
        forward_output_shape_no_batch = forward_output_shape[1:]
        backward_output_shape_no_batch = backward_output_shape[1:]

        # Now, use these shapes as needed
        # For example, write variables to self.stack_vars
        self.stack_vars += 'size_t ' + layer.name + '_input_shape[] = {' + \
                           ','.join(map(str, input_shape_no_batch)) + '}; \n'
        self.stack_vars += 'size_t ' + layer.name + '_forward_output_shape[] = {' + \
                           ','.join(map(str,
                                        forward_output_shape_no_batch)) + '}; \n'
        self.stack_vars += 'size_t ' + layer.name + '_backward_output_shape[] = {' + \
                           ','.join(map(str,
                                        backward_output_shape_no_batch)) + '}; \n'
        """
        try:
            layer.forward_layer.input_shape
            layer.backward_layer.input_shape
        except Exception:
            temp_input = keras.layers.Input(shape=layer.input_shape[2:])
            layer.layer(temp_input)
            layer.forward_layer(temp_input)
            layer.backward_layer(temp_input)
        self._write_weights_layer(layer.backward_layer)
        self._write_weights_layer(layer.forward_layer)
        if layer.merge_mode:
            self._write_outputs(layer)
            self.stack_vars += f'size_t {layer.name}_num_tensors0 = 2; \n'
            if layer.merge_mode == 'concat':
                if layer.return_sequences:
                    ax = 1
                else:
                    ax = 0
                self.stack_vars += f'size_t {layer.name}_axis = {ax}; \n'
        else:
            output_names = get_layer_io_names(layer)[1][0]
            subname = layer.layer.name
            self.stack_vars += f'k2c_tensor * {output_names[0]} = forward_{subname}_output; \n'
            self.stack_vars += f'k2c_tensor * {output_names[1]} = backward_{subname}_output; \n'
        """

    def _write_weights_TimeDistributed(self, layer):
        self._write_outputs(layer)
        try:
            layer.layer.input_shape
        except Exception:
            temp_input = keras.layers.Input(shape=layer.input.shape[2:], batch_size=1)
            layer.layer(temp_input)
        self._write_weights_layer(layer.layer)
        timeslice_input = np.squeeze(np.zeros(layer.layer.input.shape[1:]))
        timeslice_output = np.squeeze(np.zeros(layer.layer.output.shape[1:]))
        self._write_weights_array2c(
            timeslice_input, f'{layer.layer.name}_timeslice_input')
        self._write_weights_array2c(
            timeslice_output, f'{layer.layer.name}_timeslice_output')
        self.stack_vars += f'const size_t {layer.name}_timesteps = {layer.input.shape[1]}; \n'
        self.stack_vars += f'const size_t {layer.name}_in_offset = {np.prod(layer.input.shape[2:])}; \n'
        self.stack_vars += f'const size_t {layer.name}_out_offset = {np.prod(layer.output.shape[2:])}; \n'

    def _write_weights_Input(self, layer):
        self.stack_vars += ''

    def _write_weights_InputLayer(self, layer):
        self.stack_vars += ''

    def _write_weights_BatchNormalization(self, layer):
        center = layer.get_config()['center']
        scale = layer.get_config()['scale']
        axis_cfg = layer.get_config()['axis']
        if isinstance(axis_cfg, (list, tuple, np.ndarray)):
            axis_cfg = axis_cfg[0]
        if isinstance(layer.input, (list, tuple)):
            ndim = len(layer.input[0].shape)
        else:
            ndim = len(layer.input.shape)
        if axis_cfg < 0:
            axis_cfg = ndim + axis_cfg
        axis = axis_cfg - 1

        epsilon = layer.get_config()['epsilon']

        if center and scale:
            gamma = layer.get_weights()[0]
            beta = layer.get_weights()[1]
            mean = layer.get_weights()[2]
            variance = layer.get_weights()[3]
        elif center:
            beta = layer.get_weights()[0]
            mean = layer.get_weights()[1]
            variance = layer.get_weights()[2]
            gamma = np.ones(mean.shape)
        elif scale:
            gamma = layer.get_weights()[0]
            mean = layer.get_weights()[1]
            variance = layer.get_weights()[2]
            beta = np.zeros(mean.shape)
        else:
            mean = layer.get_weights()[0]
            variance = layer.get_weights()[1]
            beta = np.zeros(mean.shape)
            gamma = np.ones(mean.shape)

        stdev = np.sqrt(variance + epsilon)
        self._write_outputs(layer)
        self.stack_vars += 'size_t ' + layer.name + \
            '_axis = ' + str(axis) + '; \n'
        self._write_weights_array2c(mean, layer.name + '_mean')
        self._write_weights_array2c(stdev, layer.name + '_stdev')
        self._write_weights_array2c(gamma, layer.name + '_gamma')
        self._write_weights_array2c(beta, layer.name + '_beta')
        self.stack_vars += '\n\n'

    def _write_weights_LSTM(self, layer):
        units = layer.get_config()['units']
        self._write_outputs(layer)
        self.stack_vars += 'float ' + layer.name + \
                           '_fwork[' + str(8*units) + '] = {0}; \n'
        self.stack_vars += 'int ' + layer.name + '_go_backwards = ' + \
            str(int(layer.get_config()['go_backwards'])) + ';\n'
        self.stack_vars += 'int ' + layer.name + '_return_sequences = ' + \
            str(int(layer.get_config()['return_sequences'])) + ';\n'
        if layer.get_config()['stateful']:
            self.static_vars.update({layer.name + '_state': 2*units})
            self.stack_vars += 'float * ' + layer.name + '_state = ' + \
                self.function_name + '_states.' + \
                layer.name + '_state; \n'
        else:
            self.stack_vars += 'float ' + layer.name + \
                               '_state[' + str(2*units) + '] = {0}; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        recurrent_kernel = weights[1]
        if layer.get_config()['use_bias']:
            bias = weights[2]
        else:
            bias = np.zeros(4*units)
        ckernel = np.concatenate(np.split(kernel, 4, axis=1), axis=0)
        crecurrent_kernel = np.concatenate(
            np.split(recurrent_kernel, 4, axis=1), axis=0)
        self._write_weights_array2c(ckernel, layer.name + '_kernel')
        self._write_weights_array2c(
            crecurrent_kernel, layer.name + '_recurrent_kernel')
        self._write_weights_array2c(bias, layer.name + '_bias')
        self.stack_vars += '\n \n'

    def _write_weights_GRU(self, layer):
        units = layer.get_config()['units']
        self._write_outputs(layer)
        self.stack_vars += 'float ' + layer.name + \
            '_fwork[' + str(6*units) + '] = {0}; \n'
        self.stack_vars += 'int ' + layer.name + '_reset_after = ' + \
            str(int(layer.get_config()['reset_after'])) + ';\n'
        self.stack_vars += 'int ' + layer.name + '_go_backwards = ' + \
            str(int(layer.get_config()['go_backwards'])) + ';\n'
        self.stack_vars += 'int ' + layer.name + '_return_sequences = ' + \
            str(int(layer.get_config()['return_sequences'])) + ';\n'
        if layer.get_config()['stateful']:
            self.static_vars.update({layer.name + '_state': units})
            self.stack_vars += 'float * ' + layer.name + '_state = ' + \
                self.function_name + '_states.' + \
                layer.name + '_state; \n'
        else:
            self.stack_vars += 'float ' + layer.name + \
                '_state[' + str(units) + '] = {0}; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        recurrent_kernel = weights[1]
        if layer.get_config()['use_bias']:
            bias = weights[2]
            if layer.get_config()['reset_after']:
                rbias = bias[1]
                bias = bias[0]
            else:
                bias = bias
                rbias = np.zeros(3*units)
        else:
            bias = np.zeros(3*units)
            rbias = np.zeros(3*units)
        cbias = np.concatenate([bias, rbias], axis=0)
        ckernel = np.concatenate(np.split(kernel, 3, axis=1), axis=0)
        crecurrent_kernel = np.concatenate(
            np.split(recurrent_kernel, 3, axis=1), axis=0)
        self._write_weights_array2c(ckernel, layer.name + '_kernel')
        self._write_weights_array2c(crecurrent_kernel, layer.name +
                                    '_recurrent_kernel')
        self._write_weights_array2c(cbias, layer.name + '_bias')
        self.stack_vars += '\n \n'

    def _write_weights_SimpleRNN(self, layer):
        units = layer.get_config()['units']
        self._write_outputs(layer)
        self.stack_vars += 'int ' + layer.name + '_go_backwards = ' + \
            str(int(layer.get_config()['go_backwards'])) + ';\n'
        self.stack_vars += 'int ' + layer.name + '_return_sequences = ' + \
            str(int(layer.get_config()['return_sequences'])) + ';\n'
        self.stack_vars += 'float ' + layer.name + \
            '_fwork[' + str(2*units) + '] = {0}; \n'
        if layer.get_config()['stateful']:
            self.static_vars.update({layer.name + '_state': units})
            self.stack_vars += 'float * ' + layer.name + '_state = ' + \
                self.function_name + '_states.' + \
                layer.name + '_state; \n'
        else:
            self.stack_vars += 'float ' + layer.name + \
                '_state[' + str(units) + '] = {0}; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        recurrent_kernel = weights[1]
        if layer.get_config()['use_bias']:
            bias = weights[2]
        else:
            bias = np.zeros(units)
        self._write_weights_array2c(kernel, layer.name + '_kernel')
        self._write_weights_array2c(recurrent_kernel, layer.name +
                                    '_recurrent_kernel')
        self._write_weights_array2c(bias, layer.name + '_bias')
        self.stack_vars += '\n \n'

    def _write_weights_Dense(self, layer):
        self._write_outputs(layer)
        weights = layer.get_weights()
        A = weights[0]
        if layer.get_config()['use_bias']:
            b = weights[1]
        else:
            b = np.zeros(A.shape[1])  # No bias term

        self._write_weights_array2c(A, f"{layer.name}_kernel")
        self._write_weights_array2c(b, f"{layer.name}_bias")

        input_shape = layer.input.shape
        output_shape = layer.output.shape

        # Exclude the batch dimension and handle None values
        input_shape = [dim if dim is not None else 1 for dim in input_shape[1:]]

        fwork_size = np.prod(input_shape) + np.prod(output_shape[1:])
        self.stack_vars += f'float {layer.name}_fwork[{fwork_size}] = {{0}}; \n'
        self.stack_vars += '\n \n'

    def _write_weights_Conv1D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides'][0]
        dilation = layer.get_config()['dilation_rate'][0]
        kernel_size = layer.get_config()['kernel_size'][0]
        self.stack_vars += 'size_t ' + layer.name + \
            '_stride = ' + str(stride) + '; \n'
        self.stack_vars += 'size_t ' + layer.name + \
            '_dilation = ' + str(dilation) + '; \n'
        self._write_outputs(layer)
        inshp = layer.input.shape[1:]
        if padding == 'causal':
            pad_along_height = dilation*(kernel_size-1)
            pad_top = pad_along_height
            pad_bottom = 0
            self._write_weights_array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                                        layer.name + '_padded_input')
            self.stack_vars += 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
                + str(pad_bottom) + '}; \n'
            self.stack_vars += 'float ' + layer.name + '_fill = 0.0f; \n'
        elif padding == 'same':
            pad_along_height = dilation*(kernel_size-1)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            self._write_weights_array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                                        layer.name + '_padded_input')
            self.stack_vars += 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
                + str(pad_bottom) + '}; \n'
            self.stack_vars += 'float ' + layer.name + '_fill = 0.0f; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        if layer.get_config()['use_bias']:
            bias = weights[1]
        else:
            bias = np.zeros(kernel.shape[2])
        self._write_weights_array2c(kernel, layer.name + '_kernel')
        self._write_weights_array2c(bias, layer.name + '_bias')
        self.stack_vars += '\n \n'

    def _write_weights_Conv2D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides']
        dilation = layer.get_config()['dilation_rate']
        kernel_size = layer.get_config()['kernel_size']
        self.stack_vars += 'size_t ' + layer.name + \
            '_stride[2] = {' + ','.join([str(i) for i in stride]) + '}; \n'
        self.stack_vars += 'size_t ' + layer.name + \
            '_dilation[2] = {' + ','.join([str(i)
                                           for i in dilation]) + '}; \n'
        self._write_outputs(layer)
        if padding == 'same':
            inshp = layer.input.shape[1:]
            pad_along_height = dilation[0]*(kernel_size[0]-1)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            pad_along_width = dilation[1]*(kernel_size[1]-1)
            pad_left = pad_along_width//2
            pad_right = pad_along_width - pad_left
            padshp = (inshp[0]+pad_along_height,
                      inshp[1]+pad_along_width, inshp[2])
            pad = [pad_top, pad_bottom, pad_left, pad_right]
            self._write_weights_array2c(np.zeros(padshp), layer.name +
                                        '_padded_input')
            self.stack_vars += 'size_t ' + layer.name + \
                '_pad[4] = {' + ','.join([str(i) for i in pad]) + '}; \n'
            self.stack_vars += 'float ' + layer.name + '_fill = 0.0f; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        if layer.get_config()['use_bias']:
            bias = weights[1]
        else:
            bias = np.zeros(kernel.shape[3])
        self._write_weights_array2c(kernel, layer.name + '_kernel')
        self._write_weights_array2c(bias, layer.name + '_bias')
        self.stack_vars += '\n \n'

    def _write_weights_Conv3D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides']
        dilation = layer.get_config()['dilation_rate']
        kernel_size = layer.get_config()['kernel_size']
        self.stack_vars += 'size_t ' + layer.name + \
            '_stride[3] = {' + ','.join([str(i) for i in stride]) + '}; \n'
        self.stack_vars += 'size_t ' + layer.name + \
            '_dilation[3] = {' + ','.join([str(i)
                                           for i in dilation]) + '}; \n'
        self._write_outputs(layer)
        if padding == 'same':
            inshp = layer.input.shape[1:]
            pad_along_height = dilation[0]*(kernel_size[0]-1)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            pad_along_width = dilation[1]*(kernel_size[1]-1)
            pad_left = pad_along_width//2
            pad_right = pad_along_width - pad_left
            pad_along_depth = dilation[1]*(kernel_size[1]-1)
            pad_front = pad_along_depth//2
            pad_back = pad_along_depth - pad_front
            padshp = (inshp[0]+pad_along_height,
                      inshp[1]+pad_along_width,
                      inshp[2]+pad_along_depth,
                      inshp[3])
            pad = [pad_top, pad_bottom, pad_left,
                   pad_right, pad_front, pad_back]
            self._write_weights_array2c(np.zeros(padshp), layer.name +
                                        '_padded_input')
            self.stack_vars += 'size_t ' + layer.name + \
                '_pad[6] = {' + ','.join([str(i) for i in pad]) + '}; \n'
            self.stack_vars += 'float ' + layer.name + '_fill = 0.0f; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        if layer.get_config()['use_bias']:
            bias = weights[1]
        else:
            bias = np.zeros(kernel.shape[3])
        self._write_weights_array2c(kernel, layer.name + '_kernel')
        self._write_weights_array2c(bias, layer.name + '_bias')
        self.stack_vars += '\n \n'

    def _write_weights_MaxPooling1D(self, layer):
        return self._write_weights_Pooling1D(layer)

    def _write_weights_AveragePooling1D(self, layer):
        return self._write_weights_Pooling1D(layer)

    def _write_weights_Pooling1D(self, layer):
        pad = layer.get_config()['padding']
        stride = layer.get_config()['strides'][0]
        pool_size = layer.get_config()['pool_size'][0]
        self.stack_vars += 'size_t ' + layer.name + \
            '_stride = ' + str(stride) + '; \n'
        self.stack_vars += 'size_t ' + layer.name + \
            '_pool_size = ' + str(pool_size) + '; \n'
        self._write_outputs(layer)
        inshp = layer.input.shape[1:]
        outshp = layer.output.shape[1:]
        if pad == 'same':
            pad_along_height = max((outshp[0] - 1) * stride +
                                   pool_size - inshp[0], 0)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            self._write_weights_array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                                        layer.name + '_padded_input')
            self.stack_vars += 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
                + str(pad_bottom) + '}; \n'
            self.stack_vars += 'float ' + layer.name + '_fill = -HUGE_VALF; \n'
        self.stack_vars += '\n\n'

    def _write_weights_MaxPooling2D(self, layer):
        return self._write_weights_Pooling2D(layer)

    def _write_weights_AveragePooling2D(self, layer):
        return self._write_weights_Pooling2D(layer)

    def _write_weights_Pooling2D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides']
        pool_size = layer.get_config()['pool_size']
        self.stack_vars += 'size_t ' + layer.name + \
            '_stride[2] = {' + ','.join([str(i) for i in stride]) + '}; \n'
        self.stack_vars += 'size_t ' + layer.name + \
            '_pool_size[2] = {' + ','.join([str(i)
                                            for i in pool_size]) + '}; \n'
        self._write_outputs(layer)
        if padding == 'same':
            inshp = layer.input.shape[1:]
            outshp = layer.output.shape[1:]
            pad_along_height = max((outshp[0] - 1) * stride[0] +
                                   pool_size[0] - inshp[0], 0)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            pad_along_width = max((outshp[1] - 1) * stride[1] +
                                  pool_size[1] - inshp[1], 0)
            pad_left = int(pad_along_width // 2)
            pad_right = pad_along_width - pad_left
            padshp = (inshp[0]+pad_along_height,
                      inshp[1]+pad_along_width, inshp[2])
            pad = [pad_top, pad_bottom, pad_left, pad_right]
            self._write_weights_array2c(np.zeros(padshp), layer.name +
                                        '_padded_input')
            self.stack_vars += 'size_t ' + layer.name + \
                '_pad[4] = {' + ','.join([str(i) for i in pad]) + '}; \n'
            self.stack_vars += 'float ' + layer.name + '_fill = -HUGE_VALF; \n'
        self.stack_vars += '\n\n'

    def _write_weights_GlobalMaxPooling1D(self, layer):
        return self._write_weights_GlobalPooling(layer)

    def _write_weights_GlobalMaxPooling2D(self, layer):
        return self._write_weights_GlobalPooling(layer)

    def _write_weights_GlobalMaxPooling3D(self, layer):
        return self._write_weights_GlobalPooling(layer)

    def _write_weights_GlobalAveragePooling1D(self, layer):
        return self._write_weights_GlobalPooling(layer)

    def _write_weights_GlobalAveragePooling2D(self, layer):
        return self._write_weights_GlobalPooling(layer)

    def _write_weights_GlobalAveragePooling3D(self, layer):
        return self._write_weights_GlobalPooling(layer)

    def _write_weights_GlobalPooling(self, layer):
        self._write_outputs(layer)
        self.stack_vars += '\n\n'

    def _write_weights_Add(self, layer):
        return self._write_weights_Merge(layer)

    def _write_weights_Subtract(self, layer):
        return self._write_weights_Merge(layer)

    def _write_weights_Multiply(self, layer):
        return self._write_weights_Merge(layer)

    def _write_weights_Average(self, layer):
        return self._write_weights_Merge(layer)

    def _write_weights_Maximum(self, layer):
        return self._write_weights_Merge(layer)

    def _write_weights_Minimum(self, layer):
        return self._write_weights_Merge(layer)

    def _write_weights_Merge(self, layer):
        self._write_outputs(layer)
        inputs, outputs = get_layer_io_names(layer)
        if not outputs:
            return
        num_tensors = len(inputs)
        for i, outp in enumerate(outputs):
            self.stack_vars += (
                'size_t ' + layer.name + '_num_tensors' + str(i) +
                ' = ' + str(num_tensors) + '; \n'
            )
        self.stack_vars += '\n\n'

    def _write_weights_Concatenate(self, layer):
        inputs, outputs = get_layer_io_names(layer)
        if not outputs:
            return
        outshp = layer.output.shape[1:]
        num_tensors = len(inputs)
        ax = layer.get_config()['axis']
        if ax < 0:
            input_rank = len(layer.input[0].shape)
            ax += input_rank
        adjusted_axis = ax - 1  # Adjust for batch dimension
        for i, outp in enumerate(outputs):
            self.stack_vars += (
                'size_t ' + layer.name + '_num_tensors' + str(i) +
                ' = ' + str(num_tensors) + '; \n'
            )
            self.stack_vars += (
                'size_t ' + layer.name + '_axis = ' +
                str(adjusted_axis) + '; \n'
            )
            if outp not in self.model_io[1]:
                self._write_weights_array2c(np.zeros(outshp), outp + '_output')
        self.stack_vars += '\n\n'

    def _write_weights_ELU(self, layer):
        alpha = layer.get_config()['alpha']
        self.stack_vars += 'float ' + layer.name + \
            '_alpha = ' + str(alpha) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_LeakyReLU(self, layer):
        cfg = layer.get_config()
        alpha = cfg.get('negative_slope', cfg.get('alpha'))
        self.stack_vars += 'float ' + layer.name + \
            '_negative_slope = ' + str(alpha) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_ThresholdedReLU(self, layer):
        theta = layer.get_config()['theta']
        self.stack_vars += 'float ' + layer.name + \
            '_theta = ' + str(theta) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_ReLU(self, layer):
        max_value = layer.get_config()['max_value']
        negative_slope = layer.get_config()['negative_slope']
        threshold = layer.get_config()['threshold']
        if max_value is None:
            max_value = 'HUGE_VALF'
        self.stack_vars += 'float ' + layer.name + \
            '_max_value = ' + str(max_value) + '; \n'
        self.stack_vars += 'float ' + layer.name + '_negative_slope = ' + \
            str(negative_slope) + '; \n'
        self.stack_vars += 'float ' + layer.name + \
            '_threshold = ' + str(threshold) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_PReLU(self, layer):
        self._write_weights_array2c(
            layer.get_weights()[0], layer.name + '_alpha')
        self.stack_vars += '\n\n'

    def _write_weights_Reshape(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        newshp = layer.get_config()['target_shape']
        newndim = len(newshp)
        newshp = np.concatenate((newshp, np.ones(maxndim-newndim)))
        self.stack_vars += 'size_t ' + nm + \
            '_newndim = ' + str(newndim) + '; \n'
        self.stack_vars += 'size_t ' + nm + '_newshp[K2C_MAX_NDIM] = {' + \
            str(np.array2string(newshp.astype(int),
                                separator=',')[1:-1]) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_Permute(self, layer):
        self._write_outputs(layer)
        permute = np.array(layer.get_config()['dims']).astype(int) - 1
        self.stack_vars += 'size_t ' + layer.name + '_permute[' + str(permute.size) + '] = {' +\
            str(np.array2string(permute.astype(int),
                                separator=',')[1:-1]) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_RepeatVector(self, layer):
        self._write_outputs(layer)
        n = layer.get_config()['n']
        self.stack_vars += 'size_t ' + layer.name + '_n = ' + str(n) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_Dot(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        work_size = np.prod(layer.input[0].shape[1:]) + \
            np.prod(layer.input[1].shape[1:])
        axes = np.array(layer.get_config()['axes']) - 1
        self.stack_vars += 'size_t ' + nm + \
            '_axesA[1] = {' + str(axes[0]) + '}; \n'
        self.stack_vars += 'size_t ' + nm + \
            '_axesB[1] = {' + str(axes[1]) + '}; \n'
        self.stack_vars += 'size_t ' + nm + '_naxes = 1; \n'
        self.stack_vars += 'float ' + nm + \
            '_fwork[' + str(work_size) + '] = {0}; \n'
        self.stack_vars += 'int ' + nm + '_normalize = ' + \
            str(int(layer.get_config()['normalize'])) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_Embedding(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        kernel = layer.get_weights()[0]
        self._write_weights_array2c(kernel, nm+'_kernel')
        self.stack_vars += '\n\n'

    def _write_weights_UpSampling1D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        size = layer.get_config()['size']
        self.stack_vars += 'size_t ' + nm + '_size = ' + str(size) + '; \n'
        self.stack_vars += '\n\n'

    def _write_weights_UpSampling2D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        size = layer.get_config()['size']
        self.stack_vars += 'size_t ' + nm + '_size[2] = {' + str(size[0]) + \
            ',' + str(size[1]) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_UpSampling3D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        size = layer.get_config()['size']
        self.stack_vars += 'size_t ' + nm + '_size[3] = {' + str(size[0]) + \
            ',' + str(size[1]) + ',' + str(size[2]) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_Cropping1D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        crop_top = layer.get_config()['cropping'][0]
        crop_bottom = layer.get_config()['cropping'][1]
        self.stack_vars += 'size_t ' + nm + '_crop[2] = {' + str(crop_top) + ','\
            + str(crop_bottom) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_Cropping2D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        crop_top = layer.get_config()['cropping'][0][0]
        crop_bottom = layer.get_config()['cropping'][0][1]
        crop_left = layer.get_config()['cropping'][1][0]
        crop_right = layer.get_config()['cropping'][1][1]
        self.stack_vars += 'size_t ' + nm + '_crop[4] = {' + str(crop_top) + ','\
            + str(crop_bottom) + ',' + str(crop_left) + \
            ',' + str(crop_right) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_Cropping3D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        crop0 = layer.get_config()['cropping'][0][0]
        crop1 = layer.get_config()['cropping'][0][1]
        crop2 = layer.get_config()['cropping'][1][0]
        crop3 = layer.get_config()['cropping'][1][1]
        crop4 = layer.get_config()['cropping'][2][0]
        crop5 = layer.get_config()['cropping'][2][1]
        self.stack_vars += 'size_t ' + nm + '_crop[6] = {' + str(crop0) + ','\
            + str(crop1) + ',' + str(crop2) + ',' + str(crop3) + \
            ',' + str(crop4) + ',' + str(crop5) + '}; \n'
        self.stack_vars += '\n\n'

    def _write_weights_ZeroPadding1D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        pad_top = layer.get_config()['padding'][0]
        pad_bottom = layer.get_config()['padding'][1]
        self.stack_vars += 'size_t ' + nm + '_pad[2] = {' + str(pad_top) + ','\
            + str(pad_bottom) + '}; \n'
        self.stack_vars += 'float ' + nm + '_fill = 0.0f; \n'
        self.stack_vars += '\n\n'

    def _write_weights_ZeroPadding2D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        pad_top = layer.get_config()['padding'][0][0]
        pad_bottom = layer.get_config()['padding'][0][1]
        pad_left = layer.get_config()['padding'][1][0]
        pad_right = layer.get_config()['padding'][1][1]
        self.stack_vars += 'size_t ' + nm + '_pad[4] = {' + str(pad_top) + ','\
            + str(pad_bottom) + ',' + str(pad_left) + \
            ',' + str(pad_right) + '}; \n'
        self.stack_vars += 'float ' + nm + '_fill = 0.0f; \n'
        self.stack_vars += '\n\n'

    def _write_weights_ZeroPadding3D(self, layer):
        nm = layer.name
        self._write_outputs(layer)
        pad0 = layer.get_config()['padding'][0][0]
        pad1 = layer.get_config()['padding'][0][1]
        pad2 = layer.get_config()['padding'][1][0]
        pad3 = layer.get_config()['padding'][1][1]
        pad4 = layer.get_config()['padding'][2][0]
        pad5 = layer.get_config()['padding'][2][1]
        self.stack_vars += 'size_t ' + nm + '_pad[6] = {' + str(pad0) + ','\
            + str(pad1) + ',' + str(pad2) + ',' + str(pad3) + \
            ',' + str(pad4) + ',' + str(pad5) + '}; \n'
        self.stack_vars += 'float ' + nm + '_fill = 0.0f; \n'
        self.stack_vars += '\n\n'

    def _write_weights_ActivityRegularization(self, layer):
        # no weights needed
        pass

    def _write_weights_SpatialDropout1D(self, layer):
        # no weights needed
        pass

    def _write_weights_SpatialDropout2D(self, layer):
        # no weights needed
        pass

    def _write_weights_SpatialDropout3D(self, layer):
        # no weights needed
        pass

    def _write_weights_Flatten(self, layer):
        _, outputs = get_layer_io_names(layer)
        for i, outp in enumerate(outputs):
            inshp = layer.input.shape[1:]
            if outp not in self.model_io[1]:
                self._write_weights_array2c(
                    np.zeros(inshp).flatten(), outp + '_output')

    def _write_weights_Activation(self, layer):
        # no weights needed
        pass

    def _write_weights_Dropout(self, layer):
        # no weights needed
        pass
