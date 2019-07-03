"""layer2c.py
This file is part of keras2c
Writes individual layers to C code
"""

# imports
import numpy as np
import keras
from keras2c.io_parsing import layer_type


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


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
