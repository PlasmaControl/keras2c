 #include <string.h>
#include "k2c_include.h"

/*
Split input tensor into one output tensor
*/

void k2c_split(k2c_tensor *output, k2c_tensor *input, size_t offset)
{
    memcpy(&output->array[0], &input->array[offset], output->numel * sizeof(output->array[0]));
}
