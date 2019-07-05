"""weights2c.py
This file is part of keras2c
Gets weights and other parameters from each layer and writes to C file
"""

# imports
import numpy as np
from keras2c.io_parsing import layer_type, get_layer_io_names, get_model_io_names
maxndim = 4


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def array2c(array, name, malloc=False):
    temp = array.flatten(order='C')
    size = array.size
    shp = array.shape
    ndim = len(shp)
    shp = np.concatenate((shp, np.ones(maxndim-ndim)))
    count = 0
    s = ''
    if malloc:
        pass
    else:
        s += 'float ' + name + '_array[' + str(size) + '] = '
        if np.max(np.abs(temp)) < 1e-16:
            s += '{' + str(0) + '}; \n'
        else:
            s += '{\n'
            for i in range(size):
                if temp[i] == np.inf:
                    s += "HUGE_VALF,"
                elif temp[i] == -np.inf:
                    s += "-HUGE_VALF,"
                else:
                    s += "{:.10e}".format(temp[i]) + ','
                count += 1
                if (count) % 4 is 0:
                    s += '\n'
            s += '}; \n'
        s += 'k2c_tensor ' + name + ' = {&' + name + \
            '_array[0],' + str(int(ndim)) + ',' + str(int(size)) + ',{' + \
            np.array2string(shp.astype(int), separator=',')[
                1:-1] + '}}; \n'
    return s


class Weights2C():

    def __init__(self, model, malloc=False):

        self.model = model
        self.model_io = get_model_io_names(self.model)
        self.malloc = malloc

    def write_weights_layer(self, layer):
        method = getattr(self, 'write_weights_' + layer_type(layer))
        return method(layer)

    def write_weights(self):
        s = ''
        for layer in self.model.layers:
            method = getattr(self, 'write_weights_' + layer_type(layer))
            s += method(layer)
        return s

    def write_outputs(self, layer):
        s = ''
        _, outputs = get_layer_io_names(layer)
        for i, outp in enumerate(outputs):
            outshp = layer.get_output_at(i).shape[1:]
            if outp not in self.model_io[1]:
                s += array2c(np.zeros(outshp), outp + '_output')
        return s

    def write_weights_Input(self, layer):
        return ''

    def write_weights_InputLayer(self, layer):
        return ''

    def write_weights_BatchNormalization(self, layer):
        center = layer.get_config()['center']
        scale = layer.get_config()['scale']
        if isinstance(layer.get_config()['axis'], (list, tuple, np.ndarray)):
            axis = layer.get_config()['axis'][0]-1
        else:
            axis = layer.get_config()['axis']-1
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
        s = self.write_outputs(layer)
        s += 'size_t ' + layer.name + '_axis = ' + str(axis) + '; \n'
        s += array2c(mean, layer.name + '_mean')
        s += array2c(stdev, layer.name + '_stdev')
        s += array2c(gamma, layer.name + '_gamma')
        s += array2c(beta, layer.name + '_beta')
        s += '\n\n'
        return s

    def write_weights_LSTM(self, layer):
        units = layer.get_config()['units']
        s = self.write_outputs(layer)
        s += 'float ' + layer.name + '_fwork[' + str(8*units) + '] = {0}; \n'
        s += 'int ' + layer.name + '_go_backwards = ' + \
            str(int(layer.get_config()['go_backwards'])) + ';\n'
        s += 'int ' + layer.name + '_return_sequences = ' + \
            str(int(layer.get_config()['return_sequences'])) + ';\n'
        s += 'float ' + layer.name + '_state[' + str(2*units) + '] = {0}; \n'

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
        s += array2c(ckernel, layer.name + '_kernel')
        s += array2c(crecurrent_kernel, layer.name + '_recurrent_kernel')
        s += array2c(bias, layer.name + '_bias')
        s += '\n \n'
        return s

    def write_weights_GRU(self, layer):
        units = layer.get_config()['units']
        s = self.write_outputs(layer)
        s += 'float ' + layer.name + '_fwork[' + str(6*units) + '] = {0}; \n'
        s += 'int ' + layer.name + '_reset_after = ' + \
            str(int(layer.get_config()['reset_after'])) + ';\n'
        s += 'int ' + layer.name + '_go_backwards = ' + \
            str(int(layer.get_config()['go_backwards'])) + ';\n'
        s += 'int ' + layer.name + '_return_sequences = ' + \
            str(int(layer.get_config()['return_sequences'])) + ';\n'
        s += 'float ' + layer.name + '_state[' + str(units) + '] = {0}; \n'

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
        s += array2c(ckernel, layer.name + '_kernel')
        s += array2c(crecurrent_kernel, layer.name + '_recurrent_kernel')
        s += array2c(cbias, layer.name + '_bias')
        s += '\n \n'
        return s

    def write_weights_SimpleRNN(self, layer):
        units = layer.get_config()['units']
        s = self.write_outputs(layer)
        s += 'int ' + layer.name + '_go_backwards = ' + \
            str(int(layer.get_config()['go_backwards'])) + ';\n'
        s += 'int ' + layer.name + '_return_sequences = ' + \
            str(int(layer.get_config()['return_sequences'])) + ';\n'
        s += 'float ' + layer.name + '_fwork[' + str(2*units) + '] = {0}; \n'
        s += 'float ' + layer.name + '_state[' + str(units) + '] = {0}; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        recurrent_kernel = weights[1]
        if layer.get_config()['use_bias']:
            bias = weights[2]
        else:
            bias = np.zeros(units)
        s += array2c(kernel, layer.name + '_kernel')
        s += array2c(recurrent_kernel, layer.name + '_recurrent_kernel')
        s += array2c(bias, layer.name + '_bias')
        s += '\n \n'
        return s

    def write_weights_Dense(self, layer):
        s = self.write_outputs(layer)
        weights = layer.get_weights()
        A = weights[0]
        if layer.get_config()['use_bias']:
            b = weights[1]
        else:
            b = np.zeros(A.shape[1])

        s += array2c(A, layer.name + '_kernel')
        s += array2c(b, layer.name + '_bias')
        s += 'float ' + layer.name + \
             '_fwork[' + str(np.prod(layer.input_shape[1:]) +
                             np.prod(A.shape)) + '] = {0}; \n'
        s += '\n \n'
        return s

    def write_weights_Conv1D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides'][0]
        dilation = layer.get_config()['dilation_rate'][0]
        kernel_size = layer.get_config()['kernel_size'][0]
        s = 'size_t ' + layer.name + '_stride = ' + str(stride) + '; \n'
        s += 'size_t ' + layer.name + '_dilation = ' + str(dilation) + '; \n'
        s += self.write_outputs(layer)
        inshp = layer.get_input_at(0).shape[1:]
        if padding == 'causal':
            pad_along_height = dilation*(kernel_size-1)
            pad_top = pad_along_height
            pad_bottom = 0
            s += array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                         layer.name + '_padded_input')
            s += 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
                + str(pad_bottom) + '}; \n'
            s += 'float ' + layer.name + '_fill = 0.0f; \n'
        elif padding == 'same':
            pad_along_height = dilation*(kernel_size-1)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            s += array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                         layer.name + '_padded_input')
            s += 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
                + str(pad_bottom) + '}; \n'
            s += 'float ' + layer.name + '_fill = 0.0f; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        if layer.get_config()['use_bias']:
            bias = weights[1]
        else:
            bias = np.zeros(kernel.shape[2])
        s += array2c(kernel, layer.name + '_kernel')
        s += array2c(bias, layer.name + '_bias')
        s += '\n \n'
        return s

    def write_weights_Conv2D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides']
        dilation = layer.get_config()['dilation_rate']
        kernel_size = layer.get_config()['kernel_size']
        s = 'size_t ' + layer.name + \
            '_stride[2] = {' + ','.join([str(i) for i in stride]) + '}; \n'
        s += 'size_t ' + layer.name + \
             '_dilation[2] = {' + ','.join([str(i)
                                            for i in dilation]) + '}; \n'
        s += self.write_outputs(layer)
        if padding == 'same':
            inshp = layer.get_input_at(0).shape[1:]
            pad_along_height = dilation[0]*(kernel_size[0]-1)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            pad_along_width = dilation[1]*(kernel_size[1]-1)
            pad_left = pad_along_width//2
            pad_right = pad_along_width - pad_left
            padshp = (inshp[0]+pad_along_height,
                      inshp[1]+pad_along_width, inshp[2])
            pad = [pad_top, pad_bottom, pad_left, pad_right]
            s += array2c(np.zeros(padshp), layer.name + '_padded_input')
            s += 'size_t ' + layer.name + \
                 '_pad[4] = {' + ','.join([str(i) for i in pad]) + '}; \n'
            s += 'float ' + layer.name + '_fill = 0.0f; \n'

        weights = layer.get_weights()
        kernel = weights[0]
        if layer.get_config()['use_bias']:
            bias = weights[1]
        else:
            bias = np.zeros(kernel.shape[3])
        s += array2c(kernel, layer.name + '_kernel')
        s += array2c(bias, layer.name + '_bias')
        s += '\n \n'
        return s

    def write_weights_MaxPooling1D(self, layer):
        return self.write_weights_Pooling1D(layer)

    def write_weights_AveragePooling1D(self, layer):
        return self.write_weights_Pooling1D(layer)

    def write_weights_Pooling1D(self, layer):
        pad = layer.get_config()['padding']
        stride = layer.get_config()['strides'][0]
        pool_size = layer.get_config()['pool_size'][0]
        s = 'size_t ' + layer.name + '_stride = ' + str(stride) + '; \n'
        s += 'size_t ' + layer.name + '_pool_size = ' + str(pool_size) + '; \n'
        s += self.write_outputs(layer)
        inshp = layer.get_input_at(0).shape[1:]
        outshp = layer.get_output_at(0).shape[1:]
        if pad == 'same':
            pad_along_height = max((outshp[0] - 1) * stride +
                                   pool_size - inshp[0], 0)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            s += array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                         layer.name + '_padded_input')
            s += 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
                + str(pad_bottom) + '}; \n'
            s += 'float ' + layer.name + '_fill = -HUGE_VALF; \n'
        s += '\n\n'
        return s

    def write_weights_MaxPooling2D(self, layer):
        return self.write_weights_Pooling2D(layer)

    def write_weights_AveragePooling2D(self, layer):
        return self.write_weights_Pooling2D(layer)

    def write_weights_Pooling2D(self, layer):
        padding = layer.get_config()['padding']
        stride = layer.get_config()['strides']
        pool_size = layer.get_config()['pool_size']
        s = 'size_t ' + layer.name + \
            '_stride[2] = {' + ','.join([str(i) for i in stride]) + '}; \n'
        s += 'size_t ' + layer.name + \
             '_pool_size[2] = {' + ','.join([str(i)
                                             for i in pool_size]) + '}; \n'
        s += self.write_outputs(layer)
        if padding == 'same':
            inshp = layer.get_input_at(0).shape[1:]
            outshp = layer.get_output_at(0).shape[1:]
            pad_along_height = max((outshp[0] - 1) * stride[0] +
                                   pool_size[0] - inshp[0], 0)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
            pad_along_width = max((outshp[1] - 1) * stride[1] +
                                  pool_size[1] - inshp[1], 0)
            pad_left = pad_along_width//2
            pad_right = pad_along_width - pad_left
            padshp = (inshp[0]+pad_along_height,
                      inshp[1]+pad_along_width, inshp[2])
            pad = [pad_top, pad_bottom, pad_left, pad_right]
            s += array2c(np.zeros(padshp), layer.name + '_padded_input')
            s += 'size_t ' + layer.name + \
                '_pad[4] = {' + ','.join([str(i) for i in pad]) + '}; \n'
            s += 'float ' + layer.name + '_fill = -HUGE_VALF; \n'
        s += '\n\n'
        return s

    def write_weights_GlobalMaxPooling1D(self, layer):
        return self.write_weights_GlobalPooling(layer)

    def write_weights_GlobalMaxPooling2D(self, layer):
        return self.write_weights_GlobalPooling(layer)

    def write_weights_GlobalMaxPooling3D(self, layer):
        return self.write_weights_GlobalPooling(layer)

    def write_weights_GlobalAveragePooling1D(self, layer):
        return self.write_weights_GlobalPooling(layer)

    def write_weights_GlobalAveragePooling2D(self, layer):
        return self.write_weights_GlobalPooling(layer)

    def write_weights_GlobalAveragePooling3D(self, layer):
        return self.write_weights_GlobalPooling(layer)

    def write_weights_GlobalPooling(self, layer):
        s = self.write_outputs(layer)
        s += '\n\n'
        return s

    def write_weights_Add(self, layer):
        return self.write_weights_Merge(layer)

    def write_weights_Subtract(self, layer):
        return self.write_weights_Merge(layer)

    def write_weights_Multiply(self, layer):
        return self.write_weights_Merge(layer)

    def write_weights_Average(self, layer):
        return self.write_weights_Merge(layer)

    def write_weights_Maximum(self, layer):
        return self.write_weights_Merge(layer)

    def write_weights_Minimum(self, layer):
        return self.write_weights_Merge(layer)

    def write_weights_Merge(self, layer):
        inputs, outputs = get_layer_io_names(layer)
        s = ''
        for i, (inp, outp) in enumerate(zip(inputs, outputs)):
            outshp = layer.get_output_at(i).shape[1:]
            num_tensors = len(inp)
            s += 'size_t ' + layer.name + '_num_tensors' + str(i) + \
                 ' = ' + str(num_tensors) + '; \n'
        if outp not in self.model_io[1]:
            s += array2c(np.zeros(outshp), outp + '_output')
        s += '\n\n'
        return s

    def write_weights_ELU(self, layer):
        alpha = layer.get_config()['alpha']
        s = 'float ' + layer.name + '_alpha = ' + str(alpha) + '; \n'
        s += '\n\n'
        return s

    def write_weights_LeakyReLU(self, layer):
        alpha = layer.get_config()['alpha']
        s = 'float ' + layer.name + '_alpha = ' + str(alpha) + '; \n'
        s += '\n\n'
        return s

    def write_weights_ThresholdedReLU(self, layer):
        theta = layer.get_config()['theta']
        s = 'float ' + layer.name + '_theta = ' + str(theta) + '; \n'
        s += '\n\n'
        return s

    def write_weights_ReLU(self, layer):
        max_value = layer.get_config()['max_value']
        negative_slope = layer.get_config()['negative_slope']
        threshold = layer.get_config()['threshold']
        if max_value is None:
            max_value = 'HUGE_VALF'
        s = 'float ' + layer.name + '_max_value = ' + str(max_value) + '; \n'
        s += 'float ' + layer.name + '_negative_slope = ' + \
            str(negative_slope) + '; \n'
        s += 'float ' + layer.name + '_threshold = ' + str(threshold) + '; \n'
        s += '\n\n'
        return s

    def write_weights_PReLU(self, layer):
        s = array2c(layer.get_weights()[0], layer.name + '_alpha')
        s += '\n\n'
        return s

    def write_weights_Reshape(self, layer):
        nm = layer.name
        newshp = layer.get_config()['target_shape']
        newndim = len(newshp)
        newshp = np.concatenate((newshp, np.ones(maxndim-newndim)))
        s = 'size_t ' + nm + '_newndim = ' + str(newndim) + '; \n'
        s += 'size_t ' + nm + '_newshp[K2C_MAX_NDIM] = {' + \
            str(np.array2string(newshp.astype(int),
                                separator=',')[1:-1]) + '}; \n'
        s += '\n\n'
        return s

    def write_weights_Permute(self, layer):
        s = self.write_outputs(layer)
        permute = np.array(layer.get_config()['dims']).astype(int) - 1
        s += 'size_t ' + layer.name + '_permute[' + str(permute.size) + '] = {' +\
            str(np.array2string(permute.astype(int),
                                separator=',')[1:-1]) + '}; \n'
        s += '\n\n'
        return s

    def write_weights_RepeatVector(self, layer):
        s = self.write_outputs(layer)
        n = layer.get_config()['n']
        s += 'size_t ' + layer.name + '_n = ' + str(n) + '; \n'
        s += '\n\n'
        return s

    def write_weights_Dot(self, layer):
        nm = layer.name
        s = self.write_outputs(layer)
        work_size = np.prod(layer.input[0].shape[1:]) + \
            np.prod(layer.input[1].shape[1:])
        axes = np.array(layer.get_config()['axes']) - 1
        s += 'size_t ' + nm + '_axesA[1] = {' + str(axes[0]) + '}; \n'
        s += 'size_t ' + nm + '_axesB[1] = {' + str(axes[1]) + '}; \n'
        s += 'size_t ' + nm + '_naxes = 1; \n'
        s += 'float ' + nm + '_fwork[' + str(work_size) + '] = {0}; \n'
        s += 'int ' + nm + '_normalize = ' + \
            str(int(layer.get_config()['normalize'])) + '; \n'
        s += '\n\n'
        return s

    def write_weights_Embedding(self, layer):
        nm = layer.name
        s = self.write_outputs(layer)
        kernel = layer.get_weights()[0]
        s += array2c(kernel, nm+'_kernel')
        s += '\n\n'
        return s

    def write_weights_ZeroPadding1D(self, layer):
        nm = layer.name
        s = self.write_outputs(layer)
        pad_top = layer.get_config()['padding'][0]
        pad_bottom = layer.get_config()['padding'][1]
        s += 'size_t ' + nm + '_pad[2] = {' + str(pad_top) + ','\
            + str(pad_bottom) + '}; \n'
        s += 'float ' + nm + '_fill = 0.0f; \n'
        s += '\n\n'
        return s

    def write_weights_ZeroPadding2D(self, layer):
        nm = layer.name
        s = self.write_outputs(layer)
        pad_top = layer.get_config()['padding'][0][0]
        pad_bottom = layer.get_config()['padding'][0][1]
        pad_left = layer.get_config()['padding'][1][0]
        pad_right = layer.get_config()['padding'][1][1]
        s += 'size_t ' + nm + '_pad[4] = {' + str(pad_top) + ','\
            + str(pad_bottom) + ',' + str(pad_left) + \
            ',' + str(pad_right) + '}; \n'
        s += 'float ' + nm + '_fill = 0.0f; \n'
        s += '\n\n'
        return s

    def write_weights_ActivityRegularization(self, layer):
        return ''

    def write_weights_SpatialDropout1D(self, layer):
        return ''

    def write_weights_SpatialDropout2D(self, layer):
        return ''

    def write_weights_SpatialDropout3D(self, layer):
        return ''

    def write_weights_Flatten(self, layer):
        return ''

    def write_weights_Activation(self, layer):
        return ''

    def write_weights_Dropout(self, layer):
        return ''
