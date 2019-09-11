#ifndef KERAS2C_NORMALIZATION_LAYERS_H
#define KERAS2C_NORMALIZATION_LAYERS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "k2c_helper_functions.h"


void k2c_batch_norm(k2c_tensor* outputs, const k2c_tensor* inputs, const k2c_tensor* mean,
		    const k2c_tensor* stdev, const k2c_tensor* gamma, const k2c_tensor* beta,
		    const size_t axis) {
  size_t offset = 1;
  for (size_t i=axis+1; i<inputs->ndim; i++) {
    offset *= inputs->shape[i];}
  const size_t step = inputs->shape[axis];
  
  for (size_t i=0; i<inputs->numel; i++) {
    size_t idx = (i/offset)%step;
    outputs->array[i] = (inputs->array[i] - mean->array[idx]) /
      stdev->array[idx] *
      gamma->array[idx] +
      beta->array[idx];
	  }
}

#endif /* KERAS2C_NORMALIZATION_LAYERS_H */

