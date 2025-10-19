#include <string.h>
#include "k2c_include.h"

/**
 * 1D (temporal) Convolution.
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: stride length of the convolution.
 * :param dilation: dilation rate to use for dilated convolution.
 * :param activation: activation function to apply to output.
 */
void k2c_conv1d_transpose(k2c_tensor *output, const k2c_tensor *input,
                          const k2c_tensor *kernel, const k2c_tensor *bias,
                          const size_t stride, const size_t start_crop,
                          k2c_activationType *activation)
{
    memset(output->array, 0, output->numel * sizeof(output->array[0]));

    const size_t n_height = input->shape[0];
    const size_t n_channels = input->shape[1];
    const size_t k_size = kernel->shape[0];
    const size_t n_filters = kernel->shape[1];
    const size_t out_height = output->shape[0];

    const size_t ker_dim12 = n_channels * n_filters;

    size_t cs = 0;
    size_t ce = 0;
    size_t ts = 0;
    size_t ks = 0;

    for (size_t f = 0; f < n_filters; ++f)
    {
        for (size_t ch = 0; ch < n_channels; ++ch)
        {
            for (size_t t = 0; t < n_height; ++t)
            {
                ts = t * stride;
                if (ts > start_crop)
                {
                    cs = ts - start_crop;
                }
                else
                {
                    cs = 0;
                }
                if (ts + k_size - start_crop > out_height)
                {
                    ce = out_height;
                }
                else
                {
                    ce = ts + k_size - start_crop;
                }
                ks = cs - (ts - start_crop);
                for (size_t i = 0; i < ce - cs; ++i)
                {
                    output->array[(i + cs) * n_filters + f] +=
                        kernel->array[(i + ks) * ker_dim12 + f * n_channels + ch] *
                        input->array[t * n_channels + ch];
                }
            }
        }
    }
    // }

    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
}
