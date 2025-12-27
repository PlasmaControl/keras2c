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

    // changed some names for refactor clarity
    size_t output_start_idx = 0;    // cs
    size_t output_end_idx = 0;      // ce
    size_t output_raw_idx = 0;      // ts
    size_t kernel_offset = 0;       // ks

    for (size_t f = 0; f < n_filters; ++f)
    {
        for (size_t ch = 0; ch < n_channels; ++ch)
        {
            for (size_t t = 0; t < n_height; ++t)
            {
                output_raw_idx = t * stride;

                // start index
                if (output_raw_idx > start_crop)
                {
                    output_start_idx = output_raw_idx - start_crop;
                }
                else
                {
                    output_start_idx = 0;
                }

                // end index
                if (output_raw_idx + k_size - start_crop > out_height)
                {
                    output_end_idx = out_height;
                }
                else
                {
                    output_end_idx = output_raw_idx + k_size - start_crop;
                }

                kernel_offset = output_start_idx - (output_raw_idx - start_crop);

                // convolution
                for (size_t i = 0; i < output_end_idx - output_start_idx; ++i)
                {
                    output->array[(i + output_start_idx) * n_filters + f] +=
                        kernel->array[(i + kernel_offset) * ker_dim12 + f * n_channels + ch] *
                        input->array[t * n_channels + ch];
                }
            }
        }
    }
    // }

    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
}

/**
 * 2D Transposed Convolution (Deconvolution).
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: array[2] {stride_height, stride_width}.
 * :param dilation: array[2] {dilation_height, dilation_width}.
 * (Note: Logic below assumes dilation is 1 for the optimized bounds check).
 * :param padding: array[2] {crop_top, crop_left}.
 * Amount to crop from the output (inverse of padding).
 * :param activation: activation function to apply to output.
 */
void k2c_conv2d_transpose(k2c_tensor *output, const k2c_tensor *input,
                          const k2c_tensor *kernel, const k2c_tensor *bias,
                          const size_t *stride, const size_t *dilation,
                          const size_t *padding, k2c_activationType *activation)
{
    // Initialize output memory to zero
    memset(output->array, 0, output->numel * sizeof(output->array[0]));

    // --- Dimensions ---
    const size_t in_rows     = input->shape[0];
    const size_t in_cols     = input->shape[1];
    const size_t in_channels = input->shape[2];

    // Kernel Shape: {Rows, Cols, InChannels, OutChannels} based on reference
    const size_t k_rows      = kernel->shape[0];
    const size_t k_cols      = kernel->shape[1];
    const size_t n_filters   = kernel->shape[3];

    const size_t out_rows    = output->shape[0];
    const size_t out_cols    = output->shape[1];

    // Access strides/padding from arrays
    const size_t stride_h = stride[0];
    const size_t stride_w = stride[1];
    const size_t crop_h   = padding[0];
    const size_t crop_w   = padding[1];

    // Pre-calculate dimensional steps for Kernel
    // Kernel index math: z0 * (cols*in*out) + z1 * (in*out) + q * (out) + k
    // Note: This matches the "Out-Channel Last" memory layout of the reference.
    const size_t k_step_row = kernel->shape[1] * kernel->shape[2] * kernel->shape[3];
    const size_t k_step_col = kernel->shape[2] * kernel->shape[3];
    const size_t k_step_in  = kernel->shape[3];

    // --- Window Variables ---
    // Vertical (Rows)
    size_t row_raw_idx, row_start_idx, row_end_idx, row_ker_offset;
    // Horizontal (Cols)
    size_t col_raw_idx, col_start_idx, col_end_idx, col_ker_offset;

    // Loop 1: Filters (Output Channels)
    for (size_t f = 0; f < n_filters; ++f)
    {
        // Loop 2: Input Channels
        for (size_t ch = 0; ch < in_channels; ++ch)
        {
            // Loop 3: Input Rows
            for (size_t r = 0; r < in_rows; ++r)
            {
                // === Vertical Bounds Calculation (Similar to 1D) ===
                row_raw_idx = r * stride_h;

                // Clamp Top
                if (row_raw_idx > crop_h)
                    row_start_idx = row_raw_idx - crop_h;
                else
                    row_start_idx = 0;

                // Clamp Bottom
                if (row_raw_idx + k_rows - crop_h > out_rows)
                    row_end_idx = out_rows;
                else
                    row_end_idx = row_raw_idx + k_rows - crop_h;

                // Kernel Offset (Vertical)
                row_ker_offset = row_start_idx - (row_raw_idx - crop_h);


                // Loop 4: Input Columns
                for (size_t c = 0; c < in_cols; ++c)
                {
                    // === Horizontal Bounds Calculation ===
                    col_raw_idx = c * stride_w;

                    // Clamp Left
                    if (col_raw_idx > crop_w)
                        col_start_idx = col_raw_idx - crop_w;
                    else
                        col_start_idx = 0;

                    // Clamp Right
                    if (col_raw_idx + k_cols - crop_w > out_cols)
                        col_end_idx = out_cols;
                    else
                        col_end_idx = col_raw_idx + k_cols - crop_w;

                    // Kernel Offset (Horizontal)
                    col_ker_offset = col_start_idx - (col_raw_idx - crop_w);

                    // Pre-calculate Input Value
                    // Input Index: r * (cols*ch) + c * (ch) + ch
                    float input_val = input->array[r * (in_cols * in_channels) + c * in_channels + ch];

                    // === Inner Loops (Spatial Accumulation) ===
                    // Iterating over the VALID intersection of kernel and output
                    size_t valid_h = row_end_idx - row_start_idx;
                    size_t valid_w = col_end_idx - col_start_idx;

                    for (size_t kr = 0; kr < valid_h; ++kr)
                    {
                        for (size_t kc = 0; kc < valid_w; ++kc)
                        {
                            // 1. Output Index
                            // Row: (kr + row_start_idx)
                            // Col: (kc + col_start_idx)
                            // Channel: f
                            size_t out_r = kr + row_start_idx;
                            size_t out_c = kc + col_start_idx;

                            size_t out_idx = out_r * (out_cols * n_filters) + out_c * n_filters + f;

                            // 2. Kernel Index
                            // Row: (kr + row_ker_offset)
                            // Col: (kc + col_ker_offset)
                            // InChannel: ch
                            // OutChannel: f
                            size_t k_r = kr + row_ker_offset;
                            size_t k_c = kc + col_ker_offset;

                            size_t ker_idx = k_r * k_step_row + k_c * k_step_col + ch * k_step_in + f;

                            // 3. Accumulate
                            output->array[out_idx] += kernel->array[ker_idx] * input_val;
                        }
                    }
                }
            }
        }
    }

    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
}
