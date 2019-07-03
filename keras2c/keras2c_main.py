"""keras2c_main.py
This file is part of keras2c
Converts keras model to C code
"""

# imports
import numpy as np
import keras
maxndim = 4


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


# array2c


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
    outshp = layer.get_output_at(0).shape[1:]
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

        # layer2c


def write_layer_LSTM(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    output_activation = 'k2c_' + layer.get_config()['activation']
    recurrent_activation = 'k2c_' + layer.get_config()['recurrent_activation']

    s = 'k2c_lstm(' + outputs + ',' + inputs + ',' + nm + '_state,' + pnm + '_kernel, \n\t' + \
        pnm + '_recurrent_kernel,' + pnm + '_bias,' + nm + '_fwork, \n\t' + \
        nm + '_go_backwards,' + nm + '_return_sequences, \n\t' + \
        recurrent_activation + ',' + output_activation + '); \n'
    file.write(s)


def write_layer_Dense(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    activation = 'k2c_' + layer.get_config()['activation']

    s = 'k2c_dense(' + outputs + ',' + inputs + ',' + pnm + '_kernel, \n\t' + \
        pnm + '_bias,' + activation + ',' + nm + '_fwork); \n'
    file.write(s)


def write_layer_Conv1D(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    activation = 'k2c_' + layer.get_config()['activation']
    if layer.get_config()['padding'] == 'valid':
        s = 'k2c_conv1d(' + outputs + ',' + inputs + ',' + \
            pnm + '_kernel, \n\t' + pnm + '_bias,' + nm + \
            '_stride,' + nm + '_dilation,' + activation + '); \n'
    else:
        write_layer_ZeroPad(layer, file, inputs, pnm +
                            '_padded_input', i)
        s = 'k2c_conv1d(' + outputs + ',' + pnm + '_padded_input,' + \
            pnm + '_kernel, \n\t' + pnm + '_bias,' + nm + \
            '_stride,' + nm + '_dilation,' + activation + '); \n'
    file.write(s)


def write_layer_Conv2D(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    activation = 'k2c_' + layer.get_config()['activation']
    if layer.get_config()['padding'] == 'valid':
        s = 'k2c_conv2d(' + outputs + ',' + inputs + ',' + \
            pnm + '_kernel, \n\t' + pnm + '_bias,' + nm + \
            '_stride,' + nm + '_dilation,' + activation + '); \n'
    else:
        write_layer_ZeroPad(layer, file, inputs, pnm +
                            '_padded_input', i)
        s = 'k2c_conv2d(' + outputs + ',' + pnm + '_padded_input,' + \
            pnm + '_kernel, \n\t' + pnm + '_bias,' + nm + \
            '_stride,' + nm + '_dilation,' + activation + '); \n'
    file.write(s)


def write_layer_Pooling1D(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    if 'Max' in layer_type(layer):
        s = 'k2c_maxpool1d(' + outputs + ','
    else:
        s = 'k2c_avgpool1d(' + outputs + ','

    if layer.get_config()['padding'] == 'valid':
        s += inputs + ','
    else:
        write_layer_ZeroPad(layer, file, inputs, pnm +
                            '_padded_input', i)
        s += pnm + '_padded_input,'

    s += nm + '_pool_size, \n\t' + nm + '_stride); \n'
    file.write(s)


def write_layer_Pooling2D(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    if 'Max' in layer_type(layer):
        s = 'k2c_maxpool2d(' + outputs + ','
    else:
        s = 'k2c_avgpool2d(' + outputs + ','

    if layer.get_config()['padding'] == 'valid':
        s += inputs + ','
    else:
        write_layer_ZeroPad(layer, file, inputs, pnm +
                            '_padded_input', i)
        s += pnm + '_padded_input,'

    s += nm + '_pool_size, \n\t' + nm + '_stride); \n'
    file.write(s)


def write_layer_GlobalPooling(layer, file, inputs, outputs, i):
    if 'Max' in layer_type(layer):
        s = 'k2c_global_max_pooling('
    else:
        s = 'k2c_global_avg_pooling('
    s += outputs + ',' + inputs + '); \n'
    file.write(s)


def write_layer_Merge(layer, file, inputs, outputs, i):
    nm = layer.name
    if 'Subtract' == layer_type(layer):
        s = 'k2c_subtract('
    elif 'Add' == layer_type(layer):
        s = 'k2c_add('
    elif 'Multiply' == layer_type(layer):
        s = 'k2c_multiply('
    elif 'Average' == layer_type(layer):
        s = 'k2c_average('
    elif 'Maximum' == layer_type(layer):
        s = 'k2c_max('
    elif 'Minimum' == layer_type(layer):
        s = 'k2c_min('
    s += outputs + ',' + nm + '_num_tensors' + str(i) + ','
    c = ','.join(inputs)
    s += c + '); \n'
    file.write(s)


def write_layer_GRU(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    output_activation = 'k2c_' + layer.get_config()['activation']
    recurrent_activation = 'k2c_' + layer.get_config()['recurrent_activation']

    s = 'k2c_gru(' + outputs + ',' + inputs + ',' + nm + '_state,' + pnm + '_kernel, \n\t' + \
        pnm + '_recurrent_kernel,' + pnm + '_bias,' + nm + '_fwork, \n\t' + \
        nm + '_reset_after,' + nm + '_go_backwards,' + nm + '_return_sequences, \n\t' + \
        recurrent_activation + ',' + output_activation + '); \n'
    file.write(s)


def write_layer_SimpleRNN(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    activation = 'k2c_' + layer.get_config()['activation']

    s = 'k2c_simpleRNN(' + outputs + ',' + inputs + ',' + nm + '_state,' + pnm + '_kernel, \n\t' + \
        pnm + '_recurrent_kernel,' + pnm + '_bias,' + nm + '_fwork, \n\t' + \
        nm + '_go_backwards,' + nm + '_return_sequences,' + activation + '); \n'
    file.write(s)


def write_layer_Activation(layer, file, inputs, outputs, i, is_model_input, is_model_output):
    activation = 'k2c_' + layer.get_config()['activation']
    if is_model_input:
        inp = inputs + '->'
    else:
        inp = inputs[1:] + '.'
    s = activation + '(' + inp + 'array,' + inp + 'numel); \n'
    file.write(s)
    write_dummy_layer(layer, file, inputs, outputs, i,
                      is_model_input, is_model_output)


def write_layer_AdvancedActivation(layer, file, inputs, outputs, i, is_model_input, is_model_output):
    nm = layer.name
    if is_model_input:
        inp = inputs + '->'
    else:
        inp = inputs + '.'

    if layer_type(layer) == 'LeakyReLU':
        s = 'k2c_LeakyReLU(' + inp + 'array,' + \
            inp + 'numel,' + nm + '_alpha); \n'
    if layer_type(layer) == 'PReLU':
        s = 'k2c_PReLU(' + inp + 'array,' + inp + \
            'numel,' + nm + '_alpha.array); \n'
    if layer_type(layer) == 'ELU':
        s = 'k2c_ELU(' + inp + 'array,' + inp + \
            'numel,' + nm + '_alpha); \n'
    if layer_type(layer) == 'ThresholdedReLU':
        s = 'k2c_ThresholdedReLU(' + inp + 'array,' + \
            inp + 'numel,' + nm + '_theta); \n'
    if layer_type(layer) == 'ReLU':
        s = 'k2c_ReLU(' + inp + 'array,' + inp + 'numel,' + nm + '_max_value, \n\t' + \
            nm + '_negative_slope,' + nm + '_threshold); \n'
    file.write(s)
    write_dummy_layer(layer, file, inputs, outputs, i,
                      is_model_input, is_model_output)


def write_dummy_layer(layer, file, inputs, outputs, i, is_model_input, is_model_output):
    outputs = outputs.replace("&", "")
    inputs = inputs.replace("&", "")
    if is_model_input and is_model_output:
        s = outputs + '->ndim = ' + \
            inputs + '->ndim; // copy data into output struct \n'
        s += outputs + '->numel = ' + inputs + '->numel; \n'
        s += 'memcpy(&' + outputs + '->shape,&' + inputs + \
            '->shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
        s += 'memcpy(' + outputs + '->array,' + inputs + '->array,' + \
            outputs + '->numel*sizeof(' + outputs + '->array[0])); \n'
    elif is_model_input:
        s = 'k2c_tensor ' + outputs + '; \n'
        s += outputs + '.ndim = ' + \
            inputs + '->ndim; // copy data into output struct \n'
        s += outputs + '.numel = ' + inputs + '->numel; \n'
        s += 'memcpy(' + outputs + '.shape,' + inputs + \
            '->shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
        s += outputs + '.array = &' + inputs + \
            '->array[0]; // rename for clarity \n'
    elif is_model_output:
        s = outputs + '->ndim = ' + \
            inputs + '.ndim; // copy data into output struct \n'
        s += outputs + '->numel = ' + inputs + '.numel; \n'
        s += 'memcpy(' + outputs + '->shape,' + inputs + \
            '.shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
        s += 'memcpy(' + outputs + '->array,' + inputs + '.array,' + \
            outputs + '->numel*sizeof(' + outputs + '->array[0])); \n'
    else:
        s = 'k2c_tensor ' + outputs + '; \n'
        s += outputs + '.ndim = ' + \
            inputs + '.ndim; // copy data into output struct \n'
        s += outputs + '.numel = ' + inputs + '.numel; \n'
        s += 'memcpy(' + outputs + '.shape,' + inputs + \
            '.shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
        s += outputs + '.array = &' + inputs + \
            '.array[0]; // rename for clarity \n'
    file.write(s)


def write_layer_Reshape(layer, file, inputs, outputs, i, is_model_input, is_model_output):
    nm = layer.name
    s = 'k2c_reshape(' + inputs + ',' + nm + '_newshp,' + nm + '_newndim); \n'
    file.write(s)
    write_dummy_layer(layer, file, inputs, outputs, i,
                      is_model_input, is_model_output)


def write_layer_Flatten(layer, file, inputs, outputs, i, is_model_input, is_model_output):
    s = 'k2c_flatten(' + inputs + '); \n'
    file.write(s)
    write_dummy_layer(layer, file, inputs, outputs, i,
                      is_model_input, is_model_output)


def write_layer_Permute(layer, file, inputs, outputs, i):
    s = 'k2c_permute_dims(' + outputs + ',' + inputs + \
        ',' + layer.name + '_permute); \n'
    file.write(s)


def write_layer_RepeatVector(layer, file, inputs, outputs, i):
    s = 'k2c_repeat_vector(' + outputs + ',' + inputs + \
        ',' + layer.name + '_n); \n'
    file.write(s)


def write_layer_Dot(layer, file, inputs, outputs, i):
    nm = layer.name
    s = 'k2c_dot(' + outputs + ',' + inputs[0] + ',' + inputs[1] + ',' + nm + \
        '_axesA,' + '\n\t' + nm + '_axesB,' + \
        nm + '_naxes,' + nm + '_normalize,' + nm + '_fwork); \n'
    file.write(s)


def write_layer_BatchNorm(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    s = 'k2c_batch_norm(' + outputs + ',' + inputs + ',' + pnm + '_mean,' + \
        pnm + '_stdev,' + pnm + '_gamma,' + pnm + '_beta,' + nm + '_axis); \n'
    file.write(s)


def write_layer_Embedding(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm
    s = 'k2c_embedding(' + outputs + ',' + inputs + ',' + pnm + '_kernel); \n'
    file.write(s)


def write_layer_ZeroPad(layer, file, inputs, outputs, i):
    nm = layer.name
    pnm = '&' + nm

    if layer_type(layer)[-2:] == '1D':
        s = 'k2c_pad1d('
    elif layer_type(layer)[-2:] == '2D':
        s = 'k2c_pad2d('
    elif layer_type(layer)[-2:] == '3D':
        s = 'k2c_pad3d('
    s += outputs + ',' + inputs + ',' + nm + \
        '_fill, \n\t' + nm + '_pad); \n'
    file.write(s)


def layer2c(layer, file, inputs, outputs, i, is_model_input, is_model_output):
    if layer_type(layer) == 'Dense':
        write_layer_Dense(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'LSTM':
        write_layer_LSTM(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'GRU':
        write_layer_GRU(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'SimpleRNN':
        write_layer_SimpleRNN(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Conv1D':
        write_layer_Conv1D(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Conv2D':
        write_layer_Conv2D(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['MaxPooling1D', 'AveragePooling1D']:
        write_layer_Pooling1D(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['MaxPooling2D', 'AveragePooling2D']:
        write_layer_Pooling2D(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['GlobalMaxPooling1D', 'GlobalAveragePooling1D',
                               'GlobalMaxPooling2D', 'GlobalAveragePooling2D',
                               'GlobalMaxPooling3D', 'GlobalAveragePooling3D']:
        write_layer_GlobalPooling(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum']:
        write_layer_Merge(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Activation':
        write_layer_Activation(layer, file, inputs, outputs,
                               i, is_model_input, is_model_output)

    elif layer_type(layer) in ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'ReLU']:
        write_layer_AdvancedActivation(
            layer, file, inputs, outputs, i, is_model_input, is_model_output)

    elif layer_type(layer) == 'Reshape':
        write_layer_Reshape(layer, file, inputs, outputs, i,
                            is_model_input, is_model_output)

    elif layer_type(layer) == 'Flatten':
        write_layer_Flatten(layer, file, inputs, outputs, i,
                            is_model_input, is_model_output)

    elif layer_type(layer) in ['Dropout', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D', 'ActivityRegularization',
                               'GaussianNoise', 'GaussianDropout', 'AlphaDropout']:
        write_dummy_layer(layer, file, inputs, outputs, i,
                          is_model_input, is_model_output)

    elif layer_type(layer) == 'Permute':
        write_layer_Permute(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'RepeatVector':
        write_layer_RepeatVector(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Dot':
        write_layer_Dot(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['BatchNormalizationV1', 'BatchNormalization']:
        write_layer_BatchNorm(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Embedding':
        write_layer_Embedding(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['ZeroPadding1D', 'ZeroPadding2D', 'ZeroPadding3D']:
        write_layer_ZeroPad(layer, file, inputs, outputs, i)


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


# model2c
def model2c(model, file, function_name):
    model_inputs, model_outputs = get_model_io_names(model)

    s = '#include <stdio.h> \n#include <stddef.h> \n#include <math.h> \n#include <string.h> \n'
    s += '#include <stdarg.h> \n#include "k2c_include.h" \n'
    s += '\n \n'
    s += 'void ' + function_name + '('
    s_in = ['k2c_tensor* ' + in_nm + '_input' for in_nm in model_inputs]
    s += ', '.join(s_in) + ', '
    s_out = ['k2c_tensor* ' + out_nm + '_output' for out_nm in model_outputs]
    s += ', '.join(s_out) + ') { \n \n'
    file.write(s)

    print('Writing Weights')
    for layer in model.layers:
        weights2c(layer, file, [model_inputs, model_outputs])
    written_io = set(model_inputs)
    unwritten_io = set(get_all_io_names(model)) - written_io

    while len(unwritten_io) > 0:
        for layer in model.layers:
            layer_inputs, layer_outputs = get_layer_io_names(layer)
            for i, (inp, outp) in enumerate(zip(layer_inputs, layer_outputs)):
                if (set(flatten(inp)).issubset(written_io) and
                        set(flatten(outp)).issubset(unwritten_io))or \
                        layer_type(layer) == 'InputLayer':
                    print('Writing layer ', outp)
                    is_model_input = False
                    is_model_output = False
                    if isinstance(inp, list):
                        inp_nm = []
                        for j in inp:
                            if j in model_inputs:
                                inp_nm.append(j + '_input')
                                is_model_input = True
                            else:
                                inp_nm.append('&' + j + '_output')
                    else:
                        if inp in model_inputs:
                            inp_nm = inp + '_input'
                            is_model_input = True
                        else:
                            inp_nm = '&' + inp + '_output'
                    if isinstance(outp, list):
                        outp_nm = []
                        for o in outp:
                            if o in model_outputs:
                                outp_nm.append(outp + '_output')
                                is_model_output = True
                            else:
                                outp_nm.append('&' + outp + '_output')
                    else:
                        if outp in model_outputs:
                            outp_nm = outp + '_output'
                            is_model_output = True
                        else:
                            outp_nm = '&' + outp + '_output'

                    layer2c(layer, file, inp_nm, outp_nm,
                            i, is_model_input, is_model_output)
                    written_io |= set(flatten(inp))
                    written_io |= set(flatten(outp))
                    unwritten_io -= set(flatten(inp))
                    unwritten_io -= set(flatten(outp))
    file.write('\n }')


# keras2c
def k2c(model, function_name, num_tests=10):

    function_name = str(function_name)
    filename = function_name + '.h'
    if isinstance(model, str):
        model = keras.models.load_model(str(model_filepath))
    elif not isinstance(model, (keras.models.Model,
                                keras.engine.training.Model)):

        raise ValueError('Unknown model type. Model should ' +
                         'either be an instance of keras.models.Model, ' +
                         'or a filepath to a saved .h5 model')

    # check that the model can be converted
    check_model(model, function_name)
    print('All checks passed')

    file = open(filename, "x+")
    model2c(model, file, function_name)
    file.close()
    make_test_suite(model, function_name, num_tests)
    print("Done \n C code is in '" + function_name +
          ".h' and tests are in '" + function_name + "_test_suite.c'")


# checks

def is_valid_c_name(name):
    allowed_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
    allowed_starting_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    if not set(name).issubset(allowed_chars):
        return False
    if not set(name[0]).issubset(allowed_starting_chars):
        return False
    return True


def name_check(model):
    valid = True
    log = ''
    for layer in model.layers:
        if not is_valid_c_name(layer.name):
            valid = False
            log += "layer name '" + layer.name + "' is not a valid C name. \n"
    return valid, log


def layers_supported_check(model):
    core_layers = ['Dense', 'Activation', 'InputLayer', 'Input', 'Dropout',
                   'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D',
                   'ActivityRegularization', 'Flatten', 'Reshape', 'Permute',
                   'RepeatVector']
    conv_layers = ['Conv1D', 'Conv2D', 'ZeroPadding1D',
                   'ZeroPadding2D', 'ZeroPadding3D']
    pool_layers = ['MaxPooling1D', 'AveragePooling1D',
                   'MaxPooling2D', 'AveragePooling2D',
                   'GlobalMaxPooling1D', 'GlobalAveragePooling1D',
                   'GlobalMaxPooling2D', 'GlobalAveragePooling2D',
                   'GlobalMaxPooling3D', 'GlobalAveragePooling3D']
    local_layers = []
    recur_layers = ['LSTM', 'GRU', 'SimpleRNN']
    embed_layers = ['Embedding']
    merge_layers = ['Add', 'Subtract', 'Multiply',
                    'Average', 'Maximum', 'Minimum', 'Dot']
    activ_layers = ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'ReLU']
    norm_layers = ['BatchNormalizationV1', 'BatchNormalization']
    noise_layers = ['GaussianNoise', 'GaussianDropout', 'AlphaDropout']

    supported_layers = core_layers + conv_layers + pool_layers + local_layers + \
        recur_layers + embed_layers + merge_layers + \
        activ_layers + norm_layers + noise_layers
    valid = True
    log = ''
    for layer in model.layers:
        if not (layer_type(layer) in supported_layers):
            valid = False
            log += "layer type '" + \
                layer_type(layer) + "' is not supported at this time. \n"
    return valid, log


def activation_supported_check(model):
    supported_activations = ['linear', 'relu', 'softmax', 'softplus',
                             'softsign', 'relu', 'tanh', 'sigmoid',
                             'hard_sigmoid', 'exponential']
    valid = True
    log = ''
    for layer in model.layers:
        if 'activation' in layer.get_config():
            if not (layer.get_config()['activation'] in supported_activations):
                valid = False
                log += "activation type '" + layer.get_config()['activation'] + \
                    "' for layer '" + layer.name + \
                    "' is not supported at this time. \n"
        if 'recurrent_activation' in layer.get_config():
            if not (layer.get_config()['recurrent_activation'] in supported_activations):
                valid = False
                log += "recurrent activation type '" + \
                       layer.get_config()['recurrent_activation'] + \
                    "' for layer '" + layer.name + \
                    "' is not supported at this time. \n"
    return valid, log

# add check for masking


def config_supported_check(model):
    valid = True
    log = ''
    for layer in model.layers:
        if 'data_format' in layer.get_config():
            if layer.get_config()['data_format'] != 'channels_last':
                valid = False
                log += "data format '" + layer.get_config()['data_format'] +\
                       "' for layer '" + layer.name + \
                       "' is not supported at this time. \n"
        if 'return_state' in layer.get_config():
            if layer.get_config()['return_state']:
                valid = False
                log += "'return_state' option for layer '" + layer.name + \
                    "' is not supported at this time. \n"
        if 'stateful' in layer.get_config():
            if layer.get_config()['stateful']:
                valid = False
                log += "'stateful' option for layer '" + layer.name + \
                    "' is not supported at this time. \n"
        if 'shared_axes' in layer.get_config():
            if layer.get_config()['shared_axes'] is not None:
                valid = False
                log += "shared axes option for layer '" + layer.name + \
                    "' is not supported at this time. \n"
        if layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Average',
                                 'Maximum', 'Minimum']:
            inshps = layer.input_shape
            insize = [np.prod(inp[1:]) for inp in inshps]
            if len(set(insize)) > 1:
                valid = False
                log += "broadcasting merge functions between tensors" + \
                       " of different shapes for layer '" + \
                       layer.name + "' is not currently supported. \n"
        if layer_type(layer) in ['BatchNormalizationV1', 'BatchNormalization']:
            if isinstance(layer.get_config()['axis'], (list, tuple, np.ndarray)):
                if len(layer.get_config()['axis']) > 1:
                    valid = False
                    log += 'batch normalization along multiple axes is' + \
                           ' not currently supported. \n'
    return valid, log


def check_model(model, function_name):
    valid_fname = True
    log = 'The following errors were found: \n'
    if not is_valid_c_name(function_name):
        valid_fname = False
        log += "function name '" + function_name + "' is not a valid C name. \n"
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

# make test suite


def make_test_suite(model, function_name, num_tests=10, tol=1e-5):
    print('Writing tests')
    input_shape = []
    # output_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)
    for i in range(num_inputs):
        input_shape.insert(i, model.inputs[i].shape[1:])
  #  for i in range(num_outputs):
  #      output_shape.insert(i, model.outputs[i].shape[1:])

    file = open(function_name + '_test_suite.c', "x+")
    s = '#include <stdio.h> \n#include <math.h> \n#include <sys/time.h> \n#include "' + \
        function_name + '.h" \n\n'
    s += 'float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);\n'
    s += 'struct timeval GetTimeStamp(); \n \n'
    file.write(s)
    s = 'int main(){\n'
    file.write(s)
    for i in range(num_tests):
        # generate random input and write to file
        ct = 0
        while True:
            rand_inputs = []
            for j, _ in enumerate(model_inputs):
                rand_input = 4*np.random.random(input_shape[j]) - 2
                rand_input = rand_input[np.newaxis, ...]
                rand_inputs.insert(j, rand_input)
                # make predictions
            outputs = model.predict(rand_inputs)
            if np.isfinite(outputs).all():
                break
            if ct > 20:
                raise Exception('Cannot find inputs to the \
                network that result in a finite output')
        for j, _ in enumerate(model_inputs):
            file.write(array2c(np.squeeze(rand_inputs[j]), 'test' + str(i+1) +
                               '_' + model_inputs[j] + '_input'))

            # write predictions
        if not isinstance(outputs, list):
            outputs = [outputs]
            for j, _ in enumerate(model_outputs):
                output = outputs[j][0, :]
                file.write(array2c(output, 'keras_' +
                                   model_outputs[j] + '_test' + str(i+1)))
                file.write(array2c(np.zeros(output.shape), 'c_' +
                                   model_outputs[j] + '_test' + str(i+1)))
    s = ' float errors[' + str(num_tests*num_outputs) + '];\n'
    s += ' size_t num_tests = ' + str(num_tests) + '; \n'
    s += 'size_t num_outputs = ' + str(num_outputs) + '; \n'
    s += ' struct timeval t1 = GetTimeStamp(); \n'
    file.write(s)
    for i in range(num_tests):
        s = function_name + '('
        for j, _ in enumerate(model_inputs):
            s += '&test' + str(i+1) + '_' + model_inputs[j] + '_input,'
        s += '\n\t'
        for j, _ in enumerate(model_outputs):
            s += '&c_' + model_outputs[j] + '_test' + str(i+1) + ','
        s = s[:-1] + '); \n'
        file.write(s)
    file.write('\n')
    s = 'struct timeval t2 = GetTimeStamp(); \n'
    s += 'typedef unsigned long long u64; \n'
    s += 'u64 t1u = t1.tv_sec*1e6 + t1.tv_usec; \n'
    s += 'u64 t2u = t2.tv_sec*1e6 + t2.tv_usec; \n'
    s += 'printf("Average time over ' + str(num_tests) + \
        ' tests: %llu us \\n", (t2u-t1u)/' + str(num_tests) + '); \n'
    file.write(s)
    for i in range(num_tests):
        for j, _ in enumerate(model_outputs):
            s = 'errors[' + str(i*num_outputs+j) + '] = maxabs(&keras_' + model_outputs[j] + '_test' + \
                str(i+1) + ',&c_' + \
                model_outputs[j] + '_test' + str(i+1) + '); \n'
            file.write(s)
    s = 'float maxerror = errors[0]; \n'
    s += 'for(size_t i=1; i< num_tests*num_outputs;i++){ \n'
    s += 'if (errors[i] > maxerror) { \n'
    s += 'maxerror = errors[i];}} \n'
    s += 'printf("Max absolute error for ' + \
        str(num_tests) + ' tests: %f \\n", maxerror);\n'
    file.write(s)
    s = 'if (maxerror > ' + str(tol) + ') { \n'
    s += 'return 1;} \n'
    s += 'return 0;\n} \n\n'
    file.write(s)
    s = """float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){ \n
    float x = 0; \n
    float y = 0; \n
    for(size_t i=0; i<tensor1->numel; i++){\n
    y = fabs(tensor1->array[i]-tensor2->array[i]);
    if (y>x) {x=y;}}
    return x;}\n\n"""
    file.write(s)
    s = """struct timeval GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv;}"""
    file.write(s)
    file.close()
