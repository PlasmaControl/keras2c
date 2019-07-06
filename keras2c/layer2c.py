"""layer2c.py
This file is part of keras2c
Writes individual layers to C code
"""

# imports
from keras2c.io_parsing import layer_type, get_model_io_names, get_all_io_names, get_layer_io_names, flatten


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "GNU GPLv3"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class Layers2C():

    def __init__(self, model, malloc):
        self.model = model
        self.model_inputs, self.model_outputs = get_model_io_names(self.model)
        self.layers = ''
        self.malloc = malloc

    def write_layers(self):
        written_io = set(self.model_inputs)
        unwritten_io = set(get_all_io_names(self.model)) - written_io
        while len(unwritten_io) > 0:
            for layer in self.model.layers:
                layer_inputs, layer_outputs = get_layer_io_names(layer)
                for i, (inp, outp) in enumerate(zip(layer_inputs, layer_outputs)):
                    if (set(flatten(inp)).issubset(written_io) and
                            set(flatten(outp)).issubset(unwritten_io))or \
                            layer_type(layer) == 'InputLayer':
                        print('Writing layer ', outp)
                        method = getattr(
                            self, 'write_layer_' + layer_type(layer))
                        method(layer, inp, outp, i)
                        written_io |= set(flatten(inp))
                        written_io |= set(flatten(outp))
                        unwritten_io -= set(flatten(inp))
                        unwritten_io -= set(flatten(outp))
        return self.layers

    def format_io_names(self, layer, inp, outp, model_io=False):
        nm = layer.name
        pnm = '&' + nm
        is_model_input = False
        is_model_output = False
        if isinstance(inp, list):
            inp_nm = []
            for j in inp:
                if j in self.model_inputs:
                    inp_nm.append(j + '_input')
                    is_model_input = True
                else:
                    inp_nm.append('&' + j + '_output')
        else:
            if inp in self.model_inputs:
                inp_nm = inp + '_input'
                is_model_input = True
            else:
                inp_nm = '&' + inp + '_output'
        if isinstance(outp, list):
            outp_nm = []
            for o in outp:
                if o in self.model_outputs:
                    outp_nm.append(outp + '_output')
                    is_model_output = True
                else:
                    outp_nm.append('&' + outp + '_output')
        else:
            if outp in self.model_outputs:
                outp_nm = outp + '_output'
                is_model_output = True
            else:
                outp_nm = '&' + outp + '_output'
        if model_io:
            return nm, pnm, inp_nm, outp_nm, is_model_input, is_model_output
        else:
            return nm, pnm, inp_nm, outp_nm

    def write_layer_LSTM(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_lstm(' + outputs + ',' + inputs + ',' + nm + \
                       '_state,' + pnm + '_kernel, \n\t' + pnm + \
                       '_recurrent_kernel,' + pnm + '_bias,' + nm + \
                       '_fwork, \n\t' + nm + '_go_backwards,' + nm + \
                       '_return_sequences, \n\t' + \
                       'k2c_' + layer.get_config()['recurrent_activation'] + \
                       ',' + 'k2c_' + \
            layer.get_config()['activation'] + '); \n'

    def write_layer_Dense(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        activation = 'k2c_' + layer.get_config()['activation']

        self.layers += 'k2c_dense(' + outputs + ',' + inputs + ',' + pnm + \
            '_kernel, \n\t' + pnm + '_bias,' + activation + ',' + \
            nm + '_fwork); \n'

    def write_layer_Conv(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        activation = 'k2c_' + layer.get_config()['activation']
        if layer_type(layer)[-2:] == '1D':
            fname = 'k2c_conv1d('
        elif layer_type(layer)[-2:] == '2D':
            fname = 'k2c_conv2d('
        if layer.get_config()['padding'] == 'valid':
            self.layers += fname + outputs + ',' + inputs + ',' + \
                pnm + '_kernel, \n\t' + pnm + '_bias,' + nm + \
                '_stride,' + nm + '_dilation,' + activation + '); \n'
        else:
            self.write_layer_ZeroPad(layer, inputs, pnm +
                                     '_padded_input', i)
            self.layers += fname + outputs + ',' + pnm + \
                '_padded_input,' + pnm + '_kernel, \n\t' + \
                pnm + '_bias,' + nm + '_stride,' + nm + \
                '_dilation,' + activation + '); \n'

    def write_layer_Conv1D(self, layer, inputs, outputs, i):
        self.write_layer_Conv(layer, inputs, outputs, i)

    def write_layer_Conv2D(self, layer, inputs, outputs, i):
        self.write_layer_Conv(layer, inputs, outputs, i)

    def write_layer_MaxPooling1D(self, layer, inputs, outputs, i):
        self.write_layer_Pooling(layer, inputs, outputs, i)

    def write_layer_AveragePooling1D(self, layer, inputs, outputs, i):
        self.write_layer_Pooling(layer, inputs, outputs, i)

    def write_layer_Pooling(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        if 'Max' in layer_type(layer):
            s = 'k2c_maxpool'
        else:
            s = 'k2c_avgpool'
        if layer_type(layer)[-2:] == '1D':
            s += '1d(' + outputs + ','
        elif layer_type(layer)[-2:] == '2D':
            s += '2d(' + outputs + ','

        if layer.get_config()['padding'] == 'valid':
            s += inputs + ','
        else:
            self.write_layer_ZeroPad(layer, inputs, pnm +
                                     '_padded_input', i)
            s += pnm + '_padded_input,'

        s += nm + '_pool_size, \n\t' + nm + '_stride); \n'
        self.layers += s

    def write_layer_MaxPooling2D(self, layer, inputs, outputs, i):
        self.write_layer_Pooling(layer, inputs, outputs, i)

    def write_layer_AveragePooling2D(self, layer, inputs, outputs, i):
        self.write_layer_Pooling(layer, inputs, outputs, i)

    def write_layer_GlobalMaxPooling1D(self, layer, inputs, outputs, i):
        self.write_layer_GlobalPooling(layer, inputs, outputs, i)

    def write_layer_GlobalMaxPooling2D(self, layer, inputs, outputs, i):
        self.write_layer_GlobalPooling(layer, inputs, outputs, i)

    def write_layer_GlobalMaxPooling3D(self, layer, inputs, outputs, i):
        self.write_layer_GlobalPooling(layer, inputs, outputs, i)

    def write_layer_GlobalAveragePooling1D(self, layer, inputs, outputs, i):
        self.write_layer_GlobalPooling(layer, inputs, outputs, i)

    def write_layer_GlobalAveragePooling2D(self, layer, inputs, outputs, i):
        self.write_layer_GlobalPooling(layer, inputs, outputs, i)

    def write_layer_GlobalAveragePooling3D(self, layer, inputs, outputs, i):
        self.write_layer_GlobalPooling(layer, inputs, outputs, i)

    def write_layer_GlobalPooling(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        if 'Max' in layer_type(layer):
            self.layers += 'k2c_global_max_pooling('
        else:
            self.layers += 'k2c_global_avg_pooling('
        self.layers += outputs + ',' + inputs + '); \n'

    def write_layer_Add(self, layer, inputs, outputs, i):
        self.write_layer_Merge(layer, inputs, outputs, i)

    def write_layer_Subtract(self, layer, inputs, outputs, i):
        self.write_layer_Merge(layer, inputs, outputs, i)

    def write_layer_Multiply(self, layer, inputs, outputs, i):
        self.write_layer_Merge(layer, inputs, outputs, i)

    def write_layer_Maximum(self, layer, inputs, outputs, i):
        self.write_layer_Merge(layer, inputs, outputs, i)

    def write_layer_Minimum(self, layer, inputs, outputs, i):
        self.write_layer_Merge(layer, inputs, outputs, i)

    def write_layer_Average(self, layer, inputs, outputs, i):
        self.write_layer_Merge(layer, inputs, outputs, i)

    def write_layer_Merge(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        if 'Subtract' == layer_type(layer):
            self.layers += 'k2c_subtract('
        elif 'Add' == layer_type(layer):
            self.layers += 'k2c_add('
        elif 'Multiply' == layer_type(layer):
            self.layers += 'k2c_multiply('
        elif 'Average' == layer_type(layer):
            self.layers += 'k2c_average('
        elif 'Maximum' == layer_type(layer):
            self.layers += 'k2c_max('
        elif 'Minimum' == layer_type(layer):
            self.layers += 'k2c_min('
        self.layers += outputs + ',' + nm + '_num_tensors' + str(i) + ','
        c = ','.join(inputs)
        self.layers += c + '); \n'

    def write_layer_GRU(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_gru(' + outputs + ',' + inputs + ',' + \
            nm + '_state,' + pnm + '_kernel, \n\t' + \
            pnm + '_recurrent_kernel,' + pnm + '_bias,' + \
            nm + '_fwork, \n\t' + nm + '_reset_after,' + \
            nm + '_go_backwards,' + nm + '_return_sequences, \n\t' + \
            'k2c_' + layer.get_config()['recurrent_activation'] + \
            ',' + 'k2c_' + layer.get_config()['activation'] + '); \n'

    def write_layer_SimpleRNN(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_simpleRNN(' + outputs + ',' + inputs + \
            ',' + nm + '_state,' + pnm + '_kernel, \n\t' + \
            pnm + '_recurrent_kernel,' + pnm + '_bias,' + \
            nm + '_fwork, \n\t' + nm + '_go_backwards,' + \
            nm + '_return_sequences,' + 'k2c_' + \
            layer.get_config()['activation'] + '); \n'

    def write_layer_Activation(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        activation = 'k2c_' + layer.get_config()['activation']
        if is_model_input:
            inp = inputs + '->'
        else:
            inp = inputs[1:] + '.'
        self.layers += activation + '(' + inp + 'array,' + inp + 'numel); \n'
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_LeakyReLU(self, layer, inputs, outputs, i):
        self.write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def write_layer_PReLU(self, layer, inputs, outputs, i):
        self.write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def write_layer_ELU(self, layer, inputs, outputs, i):
        self.write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def write_layer_ThresholdedReLU(self, layer, inputs, outputs, i):
        self.write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def write_layer_ReLU(self, layer, inputs, outputs, i):
        self.write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def write_layer_AdvancedActivation(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        if is_model_input:
            inp = inputs + '->'
        else:
            inp = inputs + '.'

        if layer_type(layer) == 'LeakyReLU':
            self.layers += 'k2c_LeakyReLU(' + inp + 'array,' + \
                inp + 'numel,' + nm + '_alpha); \n'
        if layer_type(layer) == 'PReLU':
            self.layers += 'k2c_PReLU(' + inp + 'array,' + inp + \
                'numel,' + nm + '_alpha.array); \n'
        if layer_type(layer) == 'ELU':
            self.layers += 'k2c_ELU(' + inp + 'array,' + inp + \
                'numel,' + nm + '_alpha); \n'
        if layer_type(layer) == 'ThresholdedReLU':
            self.layers += 'k2c_ThresholdedReLU(' + inp + 'array,' + \
                inp + 'numel,' + nm + '_theta); \n'
        if layer_type(layer) == 'ReLU':
            self.layers += 'k2c_ReLU(' + inp + 'array,' + inp + \
                           'numel,' + nm + '_max_value, \n\t' + \
                           nm + '_negative_slope,' + nm + '_threshold); \n'
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_dummy_layer(self, layer, inputs, outputs, i, is_model_input, is_model_output):
        outputs = outputs.replace("&", "")
        inputs = inputs.replace("&", "")
        if is_model_input and is_model_output:
            self.layers += outputs + '->ndim = ' + \
                inputs + '->ndim; // copy data into output struct \n'
            self.layers += outputs + '->numel = ' + inputs + '->numel; \n'
            self.layers += 'memcpy(&' + outputs + '->shape,&' + inputs + \
                '->shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += 'memcpy(' + outputs + '->array,' + inputs + '->array,' + \
                           outputs + \
                           '->numel*sizeof(' + outputs + '->array[0])); \n'
        elif is_model_input:
            self.layers += 'k2c_tensor ' + outputs + '; \n'
            self.layers += outputs + '.ndim = ' + \
                inputs + '->ndim; // copy data into output struct \n'
            self.layers += outputs + '.numel = ' + inputs + '->numel; \n'
            self.layers += 'memcpy(' + outputs + '.shape,' + inputs + \
                '->shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += outputs + '.array = &' + inputs + \
                '->array[0]; // rename for clarity \n'
        elif is_model_output:
            self.layers += outputs + '->ndim = ' + \
                inputs + '.ndim; // copy data into output struct \n'
            self.layers += outputs + '->numel = ' + inputs + '.numel; \n'
            self.layers += 'memcpy(' + outputs + '->shape,' + inputs + \
                '.shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += 'memcpy(' + outputs + '->array,' + inputs + '.array,' + \
                           outputs + \
                           '->numel*sizeof(' + outputs + '->array[0])); \n'
        else:
            self.layers += 'k2c_tensor ' + outputs + '; \n'
            self.layers += outputs + '.ndim = ' + \
                inputs + '.ndim; // copy data into output struct \n'
            self.layers += outputs + '.numel = ' + inputs + '.numel; \n'
            self.layers += 'memcpy(' + outputs + '.shape,' + inputs + \
                '.shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += outputs + '.array = &' + inputs + \
                '.array[0]; // rename for clarity \n'

    def write_layer_Reshape(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.layers += 'k2c_reshape(' + inputs + ',' + nm + \
            '_newshp,' + nm + '_newndim); \n'
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_Flatten(self, layer, inputs, outputs, i):
        nm, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.layers += 'k2c_flatten(' + inputs + '); \n'
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_Permute(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_permute_dims(' + outputs + ',' + inputs + \
            ',' + nm + '_permute); \n'

    def write_layer_RepeatVector(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_repeat_vector(' + outputs + ',' + inputs + \
            ',' + nm + '_n); \n'

    def write_layer_Dot(self, layer, inputs, outputs, i):
        nm, _, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_dot(' + outputs + ',' + inputs[0] + \
                       ',' + inputs[1] + ',' + nm + '_axesA,' + \
                       '\n\t' + nm + '_axesB,' + nm + '_naxes,' + \
                       nm + '_normalize,' + nm + '_fwork); \n'

    def write_layer_BatchNormalization(self, layer, inputs, outputs, i):
        nm, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_batch_norm(' + outputs + ',' + inputs + \
                       ',' + pnm + '_mean,' + pnm + '_stdev,' + pnm + \
                       '_gamma,' + pnm + '_beta,' + nm + '_axis); \n'

    def write_layer_Embedding(self, layer, inputs, outputs, i):
        _, pnm, inputs, outputs = self.format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_embedding(' + outputs + ',' + inputs + \
            ',' + pnm + '_kernel); \n'

    def write_layer_ZeroPadding1D(self, layer, inputs, outputs, i):
        self.write_layer_ZeroPad(layer, inputs, outputs, i)

    def write_layer_ZeroPadding2D(self, layer, inputs, outputs, i):
        self.write_layer_ZeroPad(layer, inputs, outputs, i)

    def write_layer_ZeroPadding3D(self, layer, inputs, outputs, i):
        self.write_layer_ZeroPad(layer, inputs, outputs, i)

    def write_layer_ZeroPad(self, layer, inputs, outputs, i):
        if 'Zero' in layer_type(layer):
            nm, _, inputs, outputs = self.format_io_names(
                layer, inputs, outputs)
        else:
            nm = layer.name
        if layer_type(layer)[-2:] == '1D':
            self.layers += 'k2c_pad1d('
        elif layer_type(layer)[-2:] == '2D':
            self.layers += 'k2c_pad2d('
        elif layer_type(layer)[-2:] == '3D':
            self.layers += 'k2c_pad3d('
        self.layers += outputs + ',' + inputs + ',' + nm + \
            '_fill, \n\t' + nm + '_pad); \n'

    def write_layer_Dropout(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_SpatialDropout1D(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_SpatialDropout2D(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_SpatialDropout3D(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_ActivityRegularization(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_GaussianNoise(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_GaussianDropout(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_AlphaDropout(self, layer, inputs, outputs, i):
        _, _, inputs, outputs, is_model_input, is_model_output = self.format_io_names(
            layer, inputs, outputs, True)
        self.write_dummy_layer(layer, inputs, outputs, i,
                               is_model_input, is_model_output)

    def write_layer_Input(self, layer, inputs, outputs, i):
        self.layers += ''

    def write_layer_InputLayer(self, layer, inputs, outputs, i):
        self.layers += ''
