/**
 k2c_split_layers.c
 This file is part of keras2c
 This addition (this file) to the keras2c project was originated by Anchal Gupta
 See LICENSE file
 */

#include <string.h>
#include "k2c_include.h"

/*
Split input tensor into one output tensor
 * :param output: output tensor.
 * :param input: input tensor.
 * :param offset: The offset index at input to start copying from.
*/

void k2c_split(k2c_tensor *output, k2c_tensor *input, size_t offset)
{
    memcpy(&output->array[0], &input->array[offset], output->numel * sizeof(output->array[0]));
}