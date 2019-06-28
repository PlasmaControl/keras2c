# imports
import numpy as np
import keras
from matplotlib import pyplot as plt
import os
import copy
maxndim = 4

# array2c


def array2c(array, name):
    temp = array.flatten(order='C')
    size = array.size
    shp = array.shape
    ndim = len(shp)
    shp = np.concatenate((shp, np.ones(maxndim-ndim)))
    count = 0
    s = 'float ' + name + '_array[' + str(size) + '] = '
    if np.max(np.abs(a)) == 0:
        s += '{' + str(0) + '}; \n'
    else:
        s += '{\n'
        for i in range(size):
            if temp[i] == np.inf:
                s += "HUGE_VAL,"
            elif temp[i] == -np.inf:
                s += "-HUGE_VAL,"
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

def write_weights_LSTM(layer, file, model_io):
    weights = layer.get_weights()
    units = layer.get_config()['units']

    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]

        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))

    if layer.get_config()['go_backwards']:
        go_backwards = 1
    else:
        go_backwards = 0
    if layer.get_config()['return_sequences']:
        return_sequences = 1
    else:
        return_sequences = 0

    s = 'float ' + layer.name + '_fwork[' + str(8*units) + '] = {0}; \n'
    s += 'int ' + layer.name + '_go_backwards = ' + str(go_backwards) + ';\n'
    s += 'int ' + layer.name + '_return_sequences = ' + \
        str(return_sequences) + ';\n'
    s += 'float ' + layer.name + '_state[' + str(2*units) + '] = {0}; \n'
    file.write(s)

    kernel = weights[0]
    recurrent_kernel = weights[1]
    if layer.get_config()['use_bias']:
        bias = weights[2]
    else:
        bias = np.zeros(4*units)

    Wi = kernel[:, :units]
    Wf = kernel[:, units:2*units]
    Wc = kernel[:, 2*units:3*units]
    Wo = kernel[:, 3*units:]
    ckernel = np.concatenate([Wi, Wf, Wc, Wo], axis=0)
    Ui = recurrent_kernel[:, :units]
    Uf = recurrent_kernel[:, units:2*units]
    Uc = recurrent_kernel[:, 2*units:3*units]
    Uo = recurrent_kernel[:, 3*units:]
    crecurrent_kernel = np.concatenate([Ui, Uf, Uc, Uo], axis=0)

    file.write(array2c(ckernel, layer.name + '_kernel'))
    file.write(array2c(crecurrent_kernel, layer.name + '_recurrent_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_Dense(layer, file, model_io):
    weights = layer.get_weights()

    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]

        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))

    A = weights[0]
    if layer.get_config()['use_bias']:
        b = weights[1]
    else:
        b = np.zeros(A.shape[1])

    file.write(array2c(A, layer.name + '_kernel'))
    file.write(array2c(b, layer.name + '_bias'))
    s = 'float ' + layer.name + \
        '_fwork[' + str(insize+np.prod(A.shape)) + '] = {0}; \n'
    file.write(s)
    file.write('\n \n')


def write_weights_Conv1D(layer, file, model_io):
    weights = layer.get_weights()
    filters = weights[0]
    if layer.get_config()['use_bias']:
        bias = weights[1]
    else:
        bias = np.zeros(filters.shape[2])
    pad = layer.get_config()['padding']
    stride = layer.get_config()['strides'][0]
    dilation = layer.get_config()['dilation_rate'][0]
    kernel_size = layer.get_config()['kernel_size'][0]
    num_filters = layer.get_config()['filters']
    s = 'size_t ' + layer.name + '_stride = ' + str(stride) + '; \n'
    s += 'size_t ' + layer.name + '_dilation = ' + str(dilation) + '; \n'
    file.write(s)

    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]
        if pad == 'causal':
            pad_along_height = dilation*(kernel_size-1)
            pad_top = pad_along_height
            pad_bottom = 0
        elif pad == 'same':
            pad_along_height = max((outshp[0] - 1) * stride*dilation +
                                   kernel_size - inshp[0], 0)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
        elif pad == 'valid':
            pad_top = 0
            pad_bottom = 0

        file.write(array2c(np.zeros(inrows+pad_top+pad_bottom, incols),
                           layer.name + '_padded' + str(i) + '_input'))
        s = 'size_t ' + layer.name + '_pad' + \
            str(i) + '_top = ' + str(pad_top) + '; \n'
        s += 'size_t ' + layer.name + '_pad' + \
            str(i) + '_bottom = ' + str(pad_bottom) + '; \n'
        s += 'float ' + layer.name + '_fill' + str(i) + ' = 0; \n'
        file.write(s)
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))

    file.write(array2c(filters, layer.name + '_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_Pooling1D(layer, file, model_io):
    pad = layer.get_config()['padding']
    stride = layer.get_config()['strides'][0]
    pool_size = layer.get_config()['pool_size'][0]
    s = 'size_t ' + layer.name + '_stride = ' + str(stride) + '; \n'
    s += 'size_t ' + layer.name + '_pool_size = ' + str(pool_size) + '; \n'
    file.write(s)

    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]

        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]

        if pad == 'same':
            pad_along_height = max((outshp[0] - 1) * stride +
                                   pool_size - inshp[0], 0)
            pad_top = int(pad_along_height // 2)
            pad_bottom = int(pad_along_height - pad_top)
        elif pad == 'valid':
            pad_top = 0
            pad_bottom = 0

        file.write(array2c(np.zeros(inrows+pad_top+pad_bottom, incols),
                           layer.name + '_padded' + str(i) + '_input'))
        s = 'size_t ' + layer.name + '_pad' + \
            str(i) + '_top = ' + str(pad_top) + '; \n'
        s += 'size_t ' + layer.name + '_pad' + \
            str(i) + '_bottom = ' + str(pad_bottom) + '; \n'
        s += 'float ' + layer.name + '_fill' + str(i) + ' = -HUGE_VALF; \n'
        file.write(s)
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
    file.write('\n \n')


def write_weights_GlobalPooling1D(layer, file, model_io):
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]

        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
    file.write('\n\n')


def write_weights_Merge(layer, file, model_io):
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        outshp = layer.get_output_at(i).shape[1:]
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        numels = np.prod(outshp)
        num_tensors = len(inp)

        s = 'size_t ' + layer.name + '_num_tensors' + \
            str(i) + ' = ' + str(num_tensors) + '; \n'
        file.write(s)

        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
    file.write('\n\n')


def write_weights_GRU(layer, file, model_io):
    weights = layer.get_weights()
    units = layer.get_config()['units']
    if layer.get_config()['reset_after']:
        reset_after = 1
    else:
        reset_after = 0
    if layer.get_config()['go_backwards']:
        go_backwards = 1
    else:
        go_backwards = 0
    if layer.get_config()['return_sequences']:
        return_sequences = 1
    else:
        return_sequences = 0

    s += 'float ' + layer.name + '_fwork[' + str(6*units) + '] = {0}; \n'
    s += 'int ' + layer.name + '_reset_after = ' + str(reset_after) + ';\n'
    s += 'int ' + layer.name + '_go_backwards = ' + str(go_backwards) + ';\n'
    s += 'int ' + layer.name + '_return_sequences = ' + \
        str(return_sequences) + ';\n'
    s += 'float ' + layer.name + '_state[' + str(units) + '] = {0}; \n'
    file.write(s)

    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]

        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))

    kernel = weights[0]
    recurrent_kernel = weights[1]
    if layer.get_config()['use_bias']:
        bias = weights[2]
        if layer.get_config()['reset_after']:
            bias = b[0]
            rbias = b[1]
        else:
            bias = bias
            rbias = np.zeros(3*units)
    else:
        bias = np.zeros(3*units)
        rbias = np.zeros(3*units)
    bias = np.concatenate([bias, rbias], axis=0)
    Wz = kernel[:, :units]
    Wr = kernel[:, units:2*units]
    Wh = kernel[:, 2*units:]
    ckernel = np.concatenate([Wz, Wr, Wh], axis=0)
    Uz = recurrent_kernel[:, :units]
    Ur = recurrent_kernel[:, units:2*units]
    Uh = recurrent_kernel[:, 2*units:3*units]
    crecurrent_kernel = np.concatenate([Uz, Ur, Uh], axis=0)

    file.write(array2c(ckernel, layer.name + '_kernel'))
    file.write(array2c(crecurrent_kernel, layer.name + '_recurrent_kernel'))
    file.write(array2c(bias, layer.name + '_bias'))
    file.write('\n \n')


def write_weights_SimpleRNN(layer, file, model_io):
    weights = layer.get_weights()
    units = layer.get_config()['units']

    if layer.get_config()['go_backwards']:
        go_backwards = 1
    else:
        go_backwards = 0
    if layer.get_config()['return_sequences']:
        return_sequences = 1
    else:
        return_sequences = 0

    s = 'int ' + layer.name + '_go_backwards = ' + str(go_backwards) + ';\n'
    s += 'int ' + layer.name + '_return_sequences = ' + \
        str(return_sequences) + ';\n'
    s += 'float ' + layer.name + '_fwork[' + str(2*units) + '] = {0}; \n'
    s += 'float ' + layer.name + '_state[' + str(units) + '] = {0}; \n'
    file.write(s)

    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if len(outshp) > 1:
            outrows = outshp[0]
            outcols = outshp[1]
        else:
            outrows = 1
            outcols = outshp[0]
        if len(inshp) > 1:
            inrows = inshp[0]
            incols = inshp[1]
        else:
            inrows = 1
            incols = inshp[0]

        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))

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


def write_weights_Activation(layer, file, model_io):
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        outshp = layer.get_output_at(i).shape[1:]
        numels = np.prod(outshp)
    file.write('\n\n')


def write_weights_AdvancedActivation(layer, file, model_io):
    inputs, outputs = get_layer_io_names(layer)
    s = ''
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        outshp = layer.get_output_at(i).shape[1:]
        numels = np.prod(outshp)

    if layer_type(layer) in ['LeakyReLU', 'ELU']:
        alpha = layer.get_config()['alpha']
        s += 'float ' + layer.name + '_alpha = ' + str(alpha) + '; \n'
    if layer_type(layer) == 'ThresholdedReLU':
        theta = layer.get_config()['theta']
        s += 'float ' + layer.name + '_theta = ' + str(theta) + '; \n'
    if layer_type(layer) == 'ReLU':
        max_value = layer.get_config()['max_value']
        negative_slope = layer.get_config()['negative_slope']
        threshold = layer.get_config()['threshold']
        if max_value is None:
            max_value = 'HUGE_VAL'
        s += 'float ' + layer.name + '_max_value = ' + str(max_value) + '; \n'
        s += 'float ' + layer.name + '_negative_slope = ' + \
            str(negative_slope) + '; \n'
        s += 'float ' + layer.name + '_threshold = ' + str(threashold) + '; \n'
    if layer_type(layer) == 'PReLU':
        s += array2c(layer.get_weights()[0], layer.name + '_alpha')
    file.write(s)
    file.write('\n\n')


def write_weights_Reshape(layer, file, model_io):
    nm = layer.name
    newshp = model.layers[1].get_config()['target_shape']
    newndim = len(newshp)
    newshp = np.concatenate((shp, np.ones(maxndim-ndim)))
    s = 'size_t ' + nm + '_newndim = ' + str(newndim) + '; \n'
    s += 'size_t' + nm + '_newshp[K2C_MAX_NDIM] = {' + str(
        np.array2string(shp.astype(int), separator=',')[1:-1]) + '}; \n'
    file.write(s)


def write_weights_Permute(layer, file, model_io):
    nm = layer.name
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
        permute = np.array(layer.get_config()['dims']).astype(int) - 1
        s = 'size_t ' + nm + '_permute[' + str(permute.size) + '] = {' + str(
            np.array2string(permute.astype(int), separator=',')[1:-1]) + '}; \n'
        file.write(s)


def write_weights_RepeatVector(layer, file, model_io):
    nm = layer.name
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        inshp = layer.get_input_at(i).shape[1:]
        outshp = layer.get_output_at(i).shape[1:]
        insize = np.prod(inshp)
        outsize = np.prod(outshp)
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
        n = layer.get_config()['n']
        s = 'size_t ' + nm + '_n = ' + str(n) + '; \n'
        file.write(s)


def write_weights_Dot(layer, file, model_io):
    nm = layer.name
    inputs, outputs = get_layer_io_names(layer)
    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        work_size = np.prod(layer.get_input_at(
            i)[0].shape[1:]) + np.prod(layer.get_input_at(i)[1].shape[1:])
        if outp not in model_io[1]:
            file.write(array2c(np.zeros(outshp), outp + '_output'))
    axes = np.array(model.layers[2].get_config()['axes']) - 1
    s = 'size_t ' + nm + '_axesA[1] = {' + str(axes[0]) + '}; \n'
    s += 'size_t ' + nm + '_axesB[1] = {' + str(axes[1]) + '}; \n'
    s += 'size_t ' + nm + '_naxes = 1; \n'
    s += 'float ' + nm + '_fwork[' + str(work_size) + '] = {0}; \n'
    if layer.get_config()['normalize']:
        normalize = 1
    else:
        normalize = 0
    s += 'int ' + nm + '_normalize = ' + str(normalize) + '; \n'
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

    elif layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Maximum', 'Minimum', 'Average']:
        write_weights_Merge(layer, file, model_io)

    elif layer_type(layer) in ['MaxPooling1D', 'AveragePooling1D']:
        write_weights_Pooling1D(layer, file, model_io)

    elif layer_type(layer) in ['GlobalMaxPooling1D', 'GlobalAveragePooling']:
        write_weights_GlobalPooling1D(layer, file, model_io)

    elif layer_type(layer) == 'Activation':
        write_weights_Activation(layer, file, model_io)

    elif layer_type(layer) in ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'ReLU']:
        write_weights_AdvancedActivation(layer, file, model_io)

    elif layer_type(layer) == 'Reshape':
        write_weights_Reshape(layer, file, model_io)

    elif layer_type(layer) == 'Permute':
        write_weights_Permute(layer, file, model_io)

    elif layer_type(layer) == 'RepeatVector':
        write_weights_RepeatVector(layer, file, model_io)

    elif layer_type(layer) == 'Dot':
        write_weights_Dot(layer, file, model_io)

# layer2c


def write_layer_LSTM(layer, file, inputs, outputs, i):
    nm = layer.name
    output_activation = 'keras2c_' + layer.get_config()['activation']
    recurrent_activation = 'keras2c_' + \
        layer.get_config()['recurrent_activation']

    s = 'keras2c_lstm(' + outputs + ',' + inputs + ',' + nm + '_state,' + nm + '_kernel, \n\t' + \
        nm + '_recurrent_kernel,' + nm + '_bias,' + nm + '_fwork, \n\t' + \
        nm + '_bo_backwards,' + nm + '_return_sequences, \n\t' + \
        recurrent_activation + ',' + output_activation + '); \n'
    file.write(s)


def write_layer_Dense(layer, file, inputs, outputs, i):
    nm = layer.name
    activation = 'keras2c_' + layer.get_config()['activation']

    s = 'keras2c_dense(' + outputs + ',' + inputs + ',' + nm + '_kernel, \n\t' + \
        nm + '_bias,' + activation + ',' + nm + '_fwork); \n'
    file.write(s)


def write_layer_Conv1D(layer, file, inputs, outputs, i):
    nm = layer.name
    activation = 'keras2c_' + layer.get_config()['activation']

    s = 'keras2c_pad1d(' + nm + '_padded' + str(i) + '_input,' + inputs + ',' + nm + \
        '_fill' + str(i) + ', \n\t' + nm + '_pad' + str(i) + \
        '_top,' + nm + '_pad' + str(i) + '_bottom); \n'
    file.write(s)
    s = 'keras2c_conv1d(' + outputs + ',' + nm + '_padded' + str(i) + '_input,' + nm + '_kernel, \n\t' + \
        nm + '_bias,' + nm + '_stride,' + nm + '_dilation,' + activation + '); \n'
    file.write(s)


def write_layer_Pooling1D(layer, file, inputs, outputs, i):
    nm = layer.name
    s = 'keras2c_pad1d(' + nm + '_padded' + str(i) + '_input,' + inputs + ',' + nm + \
        '_fill' + str(i) + ', \n\t' + nm + '_pad' + str(i) + \
        '_top,' + nm + '_pad' + str(i) + '_bottom); \n'
    file.write(s)
    if 'Max' in layer_type(layer):
        s = 'keras2c_maxpool1d('
    else:
        s = 'keras2c_avgpool1d('
    s += outputs + ',' + nm + '_padded' + str(i) + '_input,' + nm + '_pool_size, \n\t' + \
        nm + '_stride); \n'
    file.write(s)


def write_layer_GlobalPooling1D(layer, file, inputs, outputs, i):
    nm = layer.name
    if 'Max' in layer_type(layer):
        s = 'keras2c_global_max_pooling_1d('
    else:
        s = 'keras2c_global_avg_pooling_1d('
    s += outputs + ',' + inputs + '); \n'
    file.write(s)


def write_layer_Merge(layer, file, inputs, outputs, i):
    nm = layer.name

    if 'Subtract' == layer_type(layer):
        s = 'keras2c_subtract('
    elif 'Add' == layer_type(layer):
        s = 'keras2c_add('
    elif 'Multiply' == layer_type(layer):
        s = 'keras2c_multiply('
    elif 'Average' == layer_type(layer):
        s = 'keras2c_average('
    elif 'Maximum' == layer_type(layer):
        s = 'keras2c_max'
    elif 'Minimum' == layer_type(layer):
        s = 'keras2c_min('
    s += outputs + ',' + nm + '_num_tensors' + str(i) + ','
    c = ','.join(inputs)
    s += c + '); \n'
    file.write(s)


def write_layer_GRU(layer, file, inputs, outputs, i):
    nm = layer.name
    output_activation = 'keras2c_' + layer.get_config()['activation']
    recurrent_activation = 'keras2c_' + \
        layer.get_config()['recurrent_activation']

    s = 'keras2c_gru(' + outputs + ',' + inputs + ',' + nm + '_state,' + nm + '_kernel, \n\t' + \
        nm + '_recurrent_kernel,' + nm + '_bias,' + nm + '_fwork, \n\t' + \
        nm + '_reset_after,' + nm + '_go_backwards,' + nm + '_return_sequences, \n\t' + \
        recurrent_activation + ',' + output_activation + '); \n'
    file.write(s)


def write_layer_SimpleRNN(layer, file, inputs, outputs, i):
    nm = layer.name
    activation = 'keras2c_' + layer.get_config()['activation']

    s = 'keras2c_simpleRNN(' + outputs + ',' + inputs + ',' + nm + '_state,' + nm + '_kernel, \n\t' + \
        nm + '_recurrent_kernel,' + nm + '_bias,' + nm + '_fwork, \n\t' + \
        nm + '_go_backwards,' + nm + '_return_sequences,' + activation + '); \n'
    file.write(s)


def write_layer_Activation(layer, file, inputs, outputs, i):
    activation = 'keras2c_' + layer.get_config()['activation']
    nm = layer.name
    s = activation + '(' + inputs + '.array,' + inputs + '.numel); \n'
    s += 'k2c_tensor *' + outputs + ' = &' + inputs + '; // rename for clarity \n'
    file.write(s)


def write_layer_AdvancedActivation(layer, file, inputs, outputs, i):
    nm = layer.name
    if layer_type(layer) == 'LeakyReLU':
        s = 'keras2c_LeakyReLU(' + inputs + '.array,' + \
            inputs + '.numel,' + nm + '_alpha); \n'
    if layer_type(layer) == 'PReLU':
        s += 'keras2c_PReLU(' + inputs + '.array,' + \
            inputs + '.numel,' + nm + '_alpha); \n'
    if layer_type(layer) == 'ELU':
        s = 'keras2c_ELU(' + inputs + '.array,' + inputs + \
            '.numel,' + nm + '_alpha); \n'
    if layer_type(layer) == 'ThresholdedReLU':
        s = 'keras2c_ThresholdedReLU(' + inputs + '.array,' + \
            inputs + '.numel,' + nm + '_theta); \n'
    if layer_type(layer) == 'ReLU':
        s = 'keras2c_ReLU(' + inputs + '.array,' + inputs + '.numel,' + nm + '_max_value, \n\t' + \
            nm + '_negative_slope,' + nm + '_threshold); \n'

    s += 'k2c_tensor *' + outputs + ' = &' + inputs + '; // rename for clarity \n'
    file.write(s)


def write_dummy_layer(layer, file, inputs, outputs, i):
    s = 'k2c_tensor *' + outputs + ' = &' + inputs + \
        '; // layer only does something during training, during predict just rename for clarity \n'
    file.write(s)


def write_layer_Reshape(layer, file, inputs, outputs, i):
    nm = layer.name
    s = 'k2c_reshape(' + inputs + ',' + nm + '_newshp,' + nm + '_newndim); \n'
    s += 'k2c_tensor *' + outputs + ' = &' + inputs + '; // rename for clarity \n'
    file.write(s)


def write_layer_Flatten(layer, file, inputs, outputs, i):
    s = 'k2c_reshape(' + inputs + '); \n'
    s += 'k2c_tensor *' + outputs + ' = &' + inputs + '; // rename for clarity \n'
    file.write(s)


def write_layer_Permute(layer, file, inputs, outputs, i):
    s = 'k2c_permute_dims(' + outputs + ',' + inputs + \
        ',' + layer.name + '_permute); \n'
    file.write(s)


def write_layer_RepeatVector(layer, file, inputs, outputs, i):
    s = 'k2c_repeat_vector(' + outputs + ',' + inputs + \
        ',' + layer.name + '_n); \n'
    file.write(s)


def write_layer_Dot(layer, file, inputs, outputs, i):
    s = 'k2c_dot(' + outputs + ',' + inputs[0] + ',' + inputs[1] + ',' + nm + '_axesA,' + nm + 'axesB,' + \
        nm + '_naxes,' + nm + '_normalize,' + nm + '_fwork); \n'
    file.write(s)


def layer2c(layer, file, inputs, outputs, i):

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

    elif layer_type(layer) in ['MaxPooling1D', 'AveragePooling1D']:
        write_layer_Pooling1D(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['GlobalMaxPooling1D', 'GlobalAveragePooling1D']:
        write_layer_GlobalPooling1D(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum']:
        write_layer_Merge(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Activation':
        write_layer_Activation(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'ReLU']:
        write_layer_AdvancedActivation(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Reshape':
        write_layer_Reshape(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Flatten':
        write_layer_Flatten(layer, file, inputs, outputs, i)

    elif layer_type(layer) in ['Dropout', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D', 'ActivityRegularization',
                               'GaussianNoise', 'GaussianDropout', 'AlphaDropout']:
        write_dummy_layer(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Permute':
        write_layer_Permute(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'RepeatVector':
        write_layer_RepeatVector(layer, file, inputs, outputs, i)

    elif layer_type(layer) == 'Dot':
        write_layer_Dot(layer, file, inputs, outputs, i)


### types, names, io

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
        except:
            error = True

    num_outputs = 0
    error = False
    while not error:
        try:
            layer.get_output_at(num_outputs)
            num_outputs += 1
        except:
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
    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)
    model_inputs, model_outputs = get_model_io_names(model)

    s = '#include <stdio.h> \n#include <stddef.h> \n#include <math.h> \n#include <string.h> \n'
    s += '#include <stdarg.h> \n#include "keras2c_include.h" \n'
    s += '\n \n'
    s += 'void ' + function_name + '('
    s_in = ['float ' + in_nm + '_input[]' for in_nm in model_inputs]
    s += ', '.join(s_in) + ', '
    s_out = ['float ' + out_nm + '_output[]' for out_nm in model_outputs]
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
                if (set(flatten(inp)).issubset(written_io) and set(flatten(outp)).issubset(unwritten_io)) \
                        or layer_type(layer) == 'InputLayer':
                    print('Writing layer ', outp)
                    if set(flatten(inp)).issubset(set(model_inputs)):
                        if isinstance(inp, list):
                            inp_nm = [nm + '_input' for nm in inp]
                        else:
                            inp_nm = inp + '_input'
                    else:
                        if isinstance(inp, list):
                            inp_nm = [nm + '_output' for nm in inp]
                        else:
                            inp_nm = inp + '_output'
                    layer2c(layer, file, inp_nm, outp + '_output', i)
                    written_io |= set(flatten(inp))
                    written_io |= set(flatten(outp))
                    unwritten_io -= set(flatten(inp))
                    unwritten_io -= set(flatten(outp))
    file.write('\n }')

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
            log += "layer name '" + layer.name + "' is not a valid C name \n"
    return valid, log


def layers_supported_check(model):
    core_layers = ['Dense', 'Activation', 'InputLayer', 'Input', 'Dropout', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D',
                   'ActivityRegularization', 'Flatten', 'Reshape', 'Permute', 'RepeatVector']
    conv_layers = ['Conv1D']
    pool_layers = ['MaxPooling1D', 'AveragePooling1D',
                   'GlobalMaxPooling1D', 'GlobalAveragePooling1D']
    local_layers = []
    recur_layers = ['LSTM', 'GRU', 'SimpleRNN']
    embed_layers = []
    merge_layers = ['Add', 'Subtract', 'Multiply',
                    'Average', 'Maximum', 'Minimum']
    activ_layers = ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'ReLU']
    norm_layers = []
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
                layer_type(layer) + "' is not supported at this time \n"
    return valid, log


def activation_supported_check(model):
    supported_activations = ['linear', 'relu', 'softmax', 'softplus', 'softsign', 'relu', 'tanh',
                             'sigmoid', 'hard_sigmoid', 'exponential']
    valid = True
    log = ''
    for layer in model.layers:
        if 'activation' in layer.get_config():
            if not (layer.get_config()['activation'] in supported_activations):
                valid = False
                log += "activation type '" + layer.get_config()['activation'] + \
                    "' for layer '" + layer.name + "' is not supported at this time \n"
        if 'recurrent_activation' in layer.get_config():
            if not (layer.get_config()['recurrent_activation'] in supported_activations):
                valid = False
                log += "recurrent activation type '" + layer.get_config()['recurrent_activation'] + \
                    "' for layer '" + layer.name + "' is not supported at this time \n"
    return valid, log

# add check for masking


def config_supported_check(model):
    valid = True
    log = ''
    for layer in model.layers:
        if 'data_format' in layer.get_config():
            if layer.get_config()['data_format'] != 'channels_last':
                valid = False
                log += "data format '" + layer.get_config()['data_format'] + "' for layer '" + \
                    layer.name + "' is not supported at this time \n"
        if 'return_state' in layer.get_config():
            if layer.get_config()['return_state']:
                valid = False
                log += "'return_state' option for layer '" + layer.name + \
                    "' is not supported at this time \n"
        if 'stateful' in layer.get_config():
            if layer.get_config()['stateful']:
                valid = False
                log += "'stateful' option for layer '" + layer.name + \
                    "' is not supported at this time \n"
        if 'shared_axes' in layer.get_config():
            if layer.get_config()['shared_axes'] is not None:
                valid = False
                log += "shared axes option for layer '" + layer.name + \
                    "' is not supported at this time"
        if layer_type(layer) in ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum']:
            inshps = layer.input_shape
            if len(set(inshps)) > 1:
                valid = False
                log += "broadcasting merge functions between tensors of different shapes for layer '" + \
                    layer.name + "' is not currently supported"
    return valid, log


def check_model(model, function_name):
    valid_fname = True
    log = 'The following errors were found: \n'
    if not is_valid_c_name(function_name):
        valid_fname = False
        log += "function name '" + function_name + "' is not a valid C name \n"
    valid_lname, name_log = name_check(model)
    log += name_log
    valid_layer, layer_log = layers_supported_check(model)
    log += layer_log
    valid_activation, activation_log = activation_supported_check(model)
    log += activation_log
    valid_config, config_log = config_supported_check(model)
    log += config_log
    assert(valid_fname and valid_lname and valid_layer and valid_activation and valid_config), log


# make test suite
def make_test_suite(model, function_name, num_tests=10):
    print('Writing tests')
    input_shape = []
    output_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)
    for i in range(num_inputs):
        input_shape.insert(i, model.inputs[i].shape[1:])
    for i in range(num_outputs):
        output_shape.insert(i, model.outputs[i].shape[1:])

    file = open(function_name + '_test_suite.c', "w+")
    s = '#include <stdio.h> \n#include <math.h> \n#include <sys/time.h> \n \n'
    s += 'float norm2(float array1[], float array2[], size_t numel);\n'
    s += 'struct timeval GetTimeStamp(); \n \n'
    file.write(s)
    fsig = 'void ' + function_name + '('
    s_in = ['float ' + in_nm + '_input[]' for in_nm in model_inputs]
    fsig += ', '.join(s_in) + ', '
    s_out = ['float ' + out_nm + '_output[]' for out_nm in model_outputs]
    fsig += ', '.join(s_out) + '); \n \n'
    file.write(fsig)

    s = 'int main(){\n'
    file.write(s)
    for i in range(num_tests):
        # generate random input and write to file
        rand_inputs = []
        for j, inpt in enumerate(model_inputs):
            rand_input = np.random.random(input_shape[j])
            file.write(array2c(rand_input, 'test' + str(i+1) +
                               '_' + model_inputs[j] + '_input'))
            rand_input = rand_input[np.newaxis, ...]
            rand_inputs.insert(j, rand_input)
        # make predictions
        outputs = model.predict(rand_inputs)
        # write predictions
        if not isinstance(outputs, list):
            outputs = [outputs]
        for j, outpt in enumerate(model_outputs):
            output = outputs[j][0, :]
            file.write(array2c(output, 'keras_' +
                               model_outputs[j] + '_test' + str(i+1)))
            s = 'float c_' + model_outputs[j] + '_test' + str(i+1) + '[' + \
                str(np.prod(output_shape[j])) + '] = {0};\n'
            file.write(s)
    s = ' float errors[' + str(num_tests*num_outputs) + '];\n'
    s += ' int num_tests = ' + str(num_tests) + '; \n'
    s += 'int num_outputs = ' + str(num_outputs) + '; \n'
    for i, outpt in enumerate(model_outputs):
        s += ' size_t numel_' + outpt + ' = ' + \
            str(np.prod(output_shape[i])) + ';\n'
    s += ' struct timeval t1 = GetTimeStamp(); \n'
    file.write(s)
    for i in range(num_tests):
        s = function_name + '('
        for j, inpt in enumerate(model_inputs):
            s += 'test' + str(i+1) + '_' + model_inputs[j] + '_input,'
        s += '\n\t'
        for j, outpt in enumerate(model_outputs):
            s += 'c_' + model_outputs[j] + '_test' + str(i+1) + ','
        s = s[:-1] + '); \n'
        file.write(s)
    file.write('\n')
    s = 'struct timeval t2 = GetTimeStamp(); \n'
    s += 'typedef unsigned long long u64; \n'
    s += 'u64 t1u = t1.tv_sec*1e6 + t1.tv_usec; \n'
    s += 'u64 t2u = t2.tv_sec*1e6 + t2.tv_usec; \n'
    s += 'printf("average time: %llu us \\n", (t2u-t1u)/' + \
        str(num_tests) + '); \n'
    file.write(s)
    for i in range(num_tests):
        for j, outpt in enumerate(model_outputs):
            s = 'errors[' + str(i*num_outputs+j) + '] = norm2(keras_' + model_outputs[j] + '_test' + \
                str(i+1) + ',c_' + \
                model_outputs[j] + '_test' + \
                str(i+1) + ',numel_' + outpt + '); \n'
            file.write(s)
    s = 'printf("L2 norm of output errors for tests:\\n");\n'
    file.write(s)
    s = 'for(size_t i=0; i< num_tests;i++){ \n'
    s += 'for(size_t j=0; j<num_outputs;j++){ \n'
    s += ' printf("test %lu, output %lu : %e \\t",i,j, errors[i*num_outputs+j]);}\n'
    s += 'printf("\\n");}\n'
    file.write(s)
    file.write('return 0;\n} \n\n')
    s = """float norm2(float array1[], float array2[], size_t numel){ \n
    float sum = 0; \n
    for(size_t i=0; i<numel; i++){\n
    sum += (array1[i]-array2[i])*(array1[i]-array2[i]);}\n
    return sqrt(sum);}\n\n"""
    file.write(s)
    s = """struct timeval GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv;}"""
    file.write(s)
    file.close()


# keras2c
def keras2c(model_filepath, function_name, num_tests=10):

    function_name = str(function_name)
    filename = function_name + '.c'
    model = keras.models.load_model(str(model_filepath))

    # check that the model can be converted
    check_model(model, function_name)
    print('All checks passed')

    file = open(filename, "w+")
    model2c(model, file, function_name)
    file.close()
    make_test_suite(model, function_name, num_tests)
    print("Done \n C code is in '" + function_name +
          ".c' and tests are in '" + function_name + "_test_suite.c'")
