#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "k2c_include.h"

void k2c_embedding(k2c_tensor* outputs, const k2c_tensor* inputs, const k2c_tensor* kernel) {

  const size_t output_dim = kernel->shape[1];
  for (size_t i = 0; i< inputs->numel; i++) {
    for (size_t j = 0; j< output_dim; j++) {
      outputs->array[i*output_dim + j] = kernel->array[(int)inputs->array[i]*output_dim+j];
    }
  }
}

