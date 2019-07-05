"""weights2c.py
This file is part of keras2c
Gets weights and other parameters from each layer and writes to C file
"""

# imports
import numpy as np
from keras2c.io_parsing import layer_type, get_layer_io_names
maxndim = 4


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def array2c(array, name):
    temp = array.flatten(order='C')
    size = array.size
    shp = array.shape
    ndim = len(shp)
    shp = np.concatenate((shp, np.ones(maxndim-ndim)))
    count = 0
    s = 'float ' + name + '_array[' + str(size) + '] = '
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
    s += 'k2c_tensor ' + name + ' = {&' + name + '_array[0],' + str(int(ndim)) + ',' + str(int(size)) + ',{' + \
        np.array2string(shp.astype(int), separator=',')[1:-1] + '}}; \n'
    return s


# weights2c

def write_outputs(layer, file, model_io):
    _, outputs = get_layer_io_names(layer)
    for i, outp in enumerate(outputs):
        outshp = layer.get_output_at(i).shape[1:]
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))


def write_weights_BatchNorm(layer, file, model_io):
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
    write_outputs(layer, file, model_io)
    s = 'size_t ' + layer.name + '_axis = ' + str(axis) + '; \n'
    file.write(s)
    file.write(array2c(mean, layer.name + '_mean'))
    file.write(array2c(stdev, layer.name + '_stdev'))
    file.write(array2c(gamma, layer.name + '_gamma'))
    file.write(array2c(beta, layer.name + '_beta'))
    file.write('\n\n')


def write_weights_LSTM(layer, file, model_io):
    units = layer.get_config()['units']
    write_outputs(layer, file, model_io)
    s = 'float ' + layer.name + '_fwork[' + str(8*units) + '] = {0}; \n'
    s += 'int ' + layer.name + '_go_backwards = ' + \
        str(int(layer.get_config()['go_backwards'])) + ';\n'
    s += 'int ' + layer.name + '_return_sequences = ' + \
        str(int(layer.get_config()['return_sequences'])) + ';\n'
    s += 'float ' + layer.name + '_state[' + str(2*units) + '] = {0}; \n'
    file.write(s)

    weights = layer.get_weights()
    kernel = weights[0]
    recurrent_kernel = weights[1]
    if layer.get_config()['use_bias']:
        bias = weights[2]
    else:
        bias = np.zeros(4*units)
    ckernel = np.concatenate([kernel[:, :units],
                              kernel[:, units:2*units],
                              kernel[:, 2*units:3*units],
                              kernel[:, 3*units:]], axis=0)
    crecurrent_kernel = np.concatenate([recurrent_kernel[:, :units],
                                        recurrent_kernel[:, units:2*units],
                                        recurrent_kernel[:, 2*units:3*units],
                                        recurrent_kernel[:, 3*units:]], axis=0)
    file.write(array2c(ckernel, layer.name + '_kernel'))
    file.write(array2c(crecurrent_kernel, layer.name + '_recurrent_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_GRU(layer, file, model_io):
    units = layer.get_config()['units']
    write_outputs(layer, file, model_io)
    s = 'float ' + layer.name + '_fwork[' + str(6*units) + '] = {0}; \n'
    s += 'int ' + layer.name + '_reset_after = ' + \
        str(int(layer.get_config()['reset_after'])) + ';\n'
    s += 'int ' + layer.name + '_go_backwards = ' + \
        str(int(layer.get_config()['go_backwards'])) + ';\n'
    s += 'int ' + layer.name + '_return_sequences = ' + \
        str(int(layer.get_config()['return_sequences'])) + ';\n'
    s += 'float ' + layer.name + '_state[' + str(units) + '] = {0}; \n'
    file.write(s)

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
    ckernel = np.concatenate([kernel[:, :units],
                              kernel[:, units:2*units],
                              kernel[:, 2*units:]], axis=0)
    crecurrent_kernel = np.concatenate([recurrent_kernel[:, :units],
                                        recurrent_kernel[:, units:2*units],
                                        recurrent_kernel[:, 2*units:3*units]], axis=0)
    file.write(array2c(ckernel, layer.name + '_kernel'))
    file.write(array2c(crecurrent_kernel, layer.name + '_recurrent_kernel'))
    file.write(array2c(cbias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_SimpleRNN(layer, file, model_io):
    units = layer.get_config()['units']
    write_outputs(layer, file, model_io)
    s = 'int ' + layer.name + '_go_backwards = ' + \
        str(int(layer.get_config()['go_backwards'])) + ';\n'
    s += 'int ' + layer.name + '_return_sequences = ' + \
        str(int(layer.get_config()['return_sequences'])) + ';\n'
    s += 'float ' + layer.name + '_fwork[' + str(2*units) + '] = {0}; \n'
    s += 'float ' + layer.name + '_state[' + str(units) + '] = {0}; \n'
    file.write(s)

    weights = layer.get_weights()
    kernel = weights[0]
    recurrent_kernel = weights[1]
    if layer.get_config()['use_bias']:
        bias = weights[2]
    else:
        bias = np.zeros(units)
    file.write(array2c(kernel, layer.name + '_kernel'))
    file.write(array2c(recurrent_kernel, layer.name + '_recurrent_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_Dense(layer, file, model_io):
    write_outputs(layer, file, model_io)
    weights = layer.get_weights()
    A = weights[0]
    if layer.get_config()['use_bias']:
        b = weights[1]
    else:
        b = np.zeros(A.shape[1])

    file.write(array2c(A, layer.name + '_kernel'))
    file.write(array2c(b, layer.name + '_bias'))
    s = 'float ' + layer.name + \
        '_fwork[' + str(np.prod(layer.input_shape[1:]) +
                        np.prod(A.shape)) + '] = {0}; \n'
    file.write(s)
    file.write('\n \n')


def write_weights_Conv1D(layer, file, model_io):
    padding = layer.get_config()['padding']
    stride = layer.get_config()['strides'][0]
    dilation = layer.get_config()['dilation_rate'][0]
    kernel_size = layer.get_config()['kernel_size'][0]
    s = 'size_t ' + layer.name + '_stride = ' + str(stride) + '; \n'
    s += 'size_t ' + layer.name + '_dilation = ' + str(dilation) + '; \n'
    file.write(s)

    write_outputs(layer, file, model_io)
    inshp = layer.get_input_at(0).shape[1:]
    if padding == 'causal':
        pad_along_height = dilation*(kernel_size-1)
        pad_top = pad_along_height
        pad_bottom = 0
        file.write(array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                           layer.name + '_padded_input'))
        s = 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
            + str(pad_bottom) + '}; \n'
        s += 'float ' + layer.name + '_fill = 0.0f; \n'
        file.write(s)
    elif padding == 'same':
        pad_along_height = dilation*(kernel_size-1)
        pad_top = int(pad_along_height // 2)
        pad_bottom = int(pad_along_height - pad_top)
        file.write(array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                           layer.name + '_padded_input'))
        s = 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
            + str(pad_bottom) + '}; \n'
        s += 'float ' + layer.name + '_fill = 0.0f; \n'
        file.write(s)

    weights = layer.get_weights()
    kernel = weights[0]
    if layer.get_config()['use_bias']:
        bias = weights[1]
    else:
        bias = np.zeros(kernel.shape[2])
    file.write(array2c(kernel, layer.name + '_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_Conv2D(layer, file, model_io):
    padding = layer.get_config()['padding']
    stride = layer.get_config()['strides']
    dilation = layer.get_config()['dilation_rate']
    kernel_size = layer.get_config()['kernel_size']
    s = 'size_t ' + layer.name + \
        '_stride[2] = {' + ','.join([str(i) for i in stride]) + '}; \n'
    s += 'size_t ' + layer.name + \
        '_dilation[2] = {' + ','.join([str(i) for i in dilation]) + '}; \n'
    file.write(s)

    write_outputs(layer, file, model_io)
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
        file.write(array2c(np.zeros(padshp), layer.name + '_padded_input'))
        s = 'size_t ' + layer.name + \
            '_pad[4] = {' + ','.join([str(i) for i in pad]) + '}; \n'
        s += 'float ' + layer.name + '_fill = 0.0f; \n'
        file.write(s)

    weights = layer.get_weights()
    kernel = weights[0]
    if layer.get_config()['use_bias']:
        bias = weights[1]
    else:
        bias = np.zeros(kernel.shape[3])
    file.write(array2c(kernel, layer.name + '_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_Pooling1D(layer, file, model_io):
    pad = layer.get_config()['padding']
    stride = layer.get_config()['strides'][0]
    pool_size = layer.get_config()['pool_size'][0]
    s = 'size_t ' + layer.name + '_stride = ' + str(stride) + '; \n'
    s += 'size_t ' + layer.name + '_pool_size = ' + str(pool_size) + '; \n'
    file.write(s)

    write_outputs(layer, file, model_io)
    inshp = layer.get_input_at(0).shape[1:]
    outshp = layer.get_output_at(0).shape[1:]
    if pad == 'same':
        pad_along_height = max((outshp[0] - 1) * stride +
                               pool_size - inshp[0], 0)
        pad_top = int(pad_along_height // 2)
        pad_bottom = int(pad_along_height - pad_top)
    elif pad == 'valid':
        pad_top = 0
        pad_bottom = 0
    file.write(array2c(np.zeros((inshp[0]+pad_top+pad_bottom, inshp[1])),
                       layer.name + '_padded_input'))
    s = 'size_t ' + layer.name + '_pad[2] = {' + str(pad_top) + ','\
        + str(pad_bottom) + '}; \n'
    s += 'float ' + layer.name + '_fill = -HUGE_VALF; \n'
    file.write(s)
    file.write('\n \n')


def write_weights_Pooling2D(layer, file, model_io):
    padding = layer.get_config()['padding']
    stride = layer.get_config()['strides']
    pool_size = layer.get_config()['pool_size']
    s = 'size_t ' + layer.name + \
        '_stride[2] = {' + ','.join([str(i) for i in stride]) + '}; \n'
    s += 'size_t ' + layer.name + \
        '_pool_size[2] = {' + ','.join([str(i) for i in pool_size]) + '}; \n'
    file.write(s)

    write_outputs(layer, file, model_io)
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
        file.write(array2c(np.zeros(padshp), layer.name + '_padded_input'))
        s = 'size_t ' + layer.name + \
            '_pad[4] = {' + ','.join([str(i) for i in pad]) + '}; \n'
        s += 'float ' + layer.name + '_fill = -HUGE_VALF; \n'
        file.write(s)


def write_weights_GlobalPooling(layer, file, model_io):
    write_outputs(layer, file, model_io)
    file.write('\n\n')


def write_weights_Merge(layer, file, model_io):
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        outshp = layer.get_output_at(i).shape[1:]
        num_tensors = len(inp)
        s = 'size_t ' + layer.name + '_num_tensors' + str(i) + \
            ' = ' + str(num_tensors) + '; \n'
        file.write(s)
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
    file.write('\n\n')


def write_weights_ELU(layer, file, model_io):
    alpha = layer.get_config()['alpha']
    s = 'float ' + layer.name + '_alpha = ' + str(alpha) + '; \n'
    file.write(s + '\n\n')


def write_weights_LeakyReLU(layer, file, model_io):
    alpha = layer.get_config()['alpha']
    s = 'float ' + layer.name + '_alpha = ' + str(alpha) + '; \n'
    file.write(s + '\n\n')


def write_weights_ThresholdedReLU(layer, file, model_io):
    theta = layer.get_config()['theta']
    s = 'float ' + layer.name + '_theta = ' + str(theta) + '; \n'
    file.write(s + '\n\n')


def write_weights_ReLU(layer, file, model_io):
    max_value = layer.get_config()['max_value']
    negative_slope = layer.get_config()['negative_slope']
    threshold = layer.get_config()['threshold']
    if max_value is None:
        max_value = 'HUGE_VALF'
    s = 'float ' + layer.name + '_max_value = ' + str(max_value) + '; \n'
    s += 'float ' + layer.name + '_negative_slope = ' + \
        str(negative_slope) + '; \n'
    s += 'float ' + layer.name + '_threshold = ' + str(threshold) + '; \n'
    file.write(s + '\n\n')


def write_weights_PReLU(layer, file, model_io):
    s = array2c(layer.get_weights()[0], layer.name + '_alpha')
    file.write(s + '\n\n')


def write_weights_Reshape(layer, file, model_io):
    nm = layer.name
    newshp = layer.get_config()['target_shape']
    newndim = len(newshp)
    newshp = np.concatenate((newshp, np.ones(maxndim-newndim)))
    s = 'size_t ' + nm + '_newndim = ' + str(newndim) + '; \n'
    s += 'size_t ' + nm + '_newshp[K2C_MAX_NDIM] = {' + \
        str(np.array2string(newshp.astype(int), separator=',')[1:-1]) + '}; \n'
    file.write(s + '\n\n')


def write_weights_Permute(layer, file, model_io):
    write_outputs(layer, file, model_io)
    permute = np.array(layer.get_config()['dims']).astype(int) - 1
    s = 'size_t ' + layer.name + '_permute[' + str(permute.size) + '] = {' +\
        str(np.array2string(permute.astype(int),
                            separator=',')[1:-1]) + '}; \n'
    file.write(s + '\n\n')


def write_weights_RepeatVector(layer, file, model_io):
    write_outputs(layer, file, model_io)
    n = layer.get_config()['n']
    s = 'size_t ' + layer.name + '_n = ' + str(n) + '; \n'
    file.write(s + '\n\n')


def write_weights_Dot(layer, file, model_io):
    nm = layer.name
    write_outputs(layer, file, model_io)
    work_size = np.prod(layer.input[0].shape[1:]) + \
        np.prod(layer.input[1].shape[1:])
    axes = np.array(layer.get_config()['axes']) - 1
    s = 'size_t ' + nm + '_axesA[1] = {' + str(axes[0]) + '}; \n'
    s += 'size_t ' + nm + '_axesB[1] = {' + str(axes[1]) + '}; \n'
    s += 'size_t ' + nm + '_naxes = 1; \n'
    s += 'float ' + nm + '_fwork[' + str(work_size) + '] = {0}; \n'
    s += 'int ' + nm + '_normalize = ' + \
        str(int(layer.get_config()['normalize'])) + '; \n'
    file.write(s + '\n\n')


def write_weights_Embedding(layer, file, model_io):
    nm = layer.name
    write_outputs(layer, file, model_io)
    kernel = layer.get_weights()[0]
    file.write(array2c(kernel, nm+'_kernel'))
    file.write('\n\n')


def write_weights_ZeroPad1D(layer, file, model_io):
    nm = layer.name
    write_outputs(layer, file, model_io)
    pad_top = layer.get_config()['padding'][0]
    pad_bottom = layer.get_config()['padding'][1]
    s = 'size_t ' + nm + '_pad[2] = {' + str(pad_top) + ','\
        + str(pad_bottom) + '}; \n'
    s += 'float ' + nm + '_fill = 0.0f; \n'
    file.write(s)


def write_weights_ZeroPad2D(layer, file, model_io):
    nm = layer.name
    write_outputs(layer, file, model_io)
    pad_top = layer.get_config()['padding'][0][0]
    pad_bottom = layer.get_config()['padding'][0][1]
    pad_left = layer.get_config()['padding'][1][0]
    pad_right = layer.get_config()['padding'][1][1]
    s = 'size_t ' + nm + '_pad[4] = {' + str(pad_top) + ','\
        + str(pad_bottom) + ',' + str(pad_left) + \
        ',' + str(pad_right) + '}; \n'
    s += 'float ' + nm + '_fill = 0.0f; \n'
    file.write(s)


def weights2c(layer, file, model_io):
    if layer_type(layer) == 'Dense':
        write_weights_Dense(layer, file, model_io)

    elif layer_type(layer) == 'LSTM':
        write_weights_LSTM(layer, file, model_io)

    elif layer_type(layer) == 'GRU':
        write_weights_GRU(layer, file, model_io)

    elif layer_type(layer) == 'SimpleRNN':
        write_weights_SimpleRNN(layer, file, model_io)

    elif layer_type(layer) == 'Conv1D':
        write_weights_Conv1D(layer, file, model_io)

    elif layer_type(layer) == 'Conv2D':
        write_weights_Conv2D(layer, file, model_io)

    elif layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Maximum', 'Minimum', 'Average']:
        write_weights_Merge(layer, file, model_io)

    elif layer_type(layer) in ['MaxPooling1D', 'AveragePooling1D']:
        write_weights_Pooling1D(layer, file, model_io)

    elif layer_type(layer) in ['MaxPooling2D', 'AveragePooling2D']:
        write_weights_Pooling2D(layer, file, model_io)

    elif layer_type(layer) in ['GlobalMaxPooling1D', 'GlobalAveragePooling1D',
                               'GlobalMaxPooling2D', 'GlobalAveragePooling2D',
                               'GlobalMaxPooling3D', 'GlobalAveragePooling3D']:
        write_weights_GlobalPooling(layer, file, model_io)

    elif layer_type(layer) == 'LeakyReLU':
        write_weights_LeakyReLU(layer, file, model_io)

    elif layer_type(layer) == 'ELU':
        write_weights_ELU(layer, file, model_io)

    elif layer_type(layer) == 'PReLU':
        write_weights_PReLU(layer, file, model_io)

    elif layer_type(layer) == 'ThresholdedReLU':
        write_weights_ThresholdedReLU(layer, file, model_io)

    elif layer_type(layer) == 'ReLU':
        write_weights_ReLU(layer, file, model_io)

    elif layer_type(layer) == 'Reshape':
        write_weights_Reshape(layer, file, model_io)

    elif layer_type(layer) == 'Permute':
        write_weights_Permute(layer, file, model_io)

    elif layer_type(layer) == 'RepeatVector':
        write_weights_RepeatVector(layer, file, model_io)

    elif layer_type(layer) == 'Dot':
        write_weights_Dot(layer, file, model_io)

    elif layer_type(layer) in ['BatchNormalizationV1', 'BatchNormalization']:
        write_weights_BatchNorm(layer, file, model_io)

    elif layer_type(layer) == 'Embedding':
        write_weights_Embedding(layer, file, model_io)

    elif layer_type(layer) == 'ZeroPadding1D':
        write_weights_ZeroPad1D(layer, file, model_io)

    elif layer_type(layer) == 'ZeroPadding2D':
        write_weights_ZeroPad2D(layer, file, model_io)
