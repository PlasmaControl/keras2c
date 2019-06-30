#ifndef KERAS2C_CONVOLUTION_LAYERS_H
#define KERAS2C_CONVOLUTION_LAYERS_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "k2c_helper_functions.h"

void k2c_pad1d(k2c_tensor* padded_input, k2c_tensor* input, float fill,
	       size_t pad_top, size_t pad_bottom) {
  /* pads array in height dimension. 
     fill = fill value
     pad_top: number of rows of fill to concatenate on top of input
     pad_bottom: number of rows of fill to cat on bottom of input */

  // pad the "top", ie, beginning of array
  size_t in_height = input->shape[0];
  size_t in_width = input->shape[1];
  
  for (size_t i=0; i<pad_top; i++) {
    size_t idx = i*in_width;
    for (size_t j=0; j<in_width; j++){
      padded_input->array[idx+j] = fill;
    }
  }
  // put the original values in the middle
  for (size_t i=pad_top, k=0; i<pad_top+in_height; i++, k++){
    size_t padidx = i*in_width;
    size_t inidx = k*in_width;
    for (size_t j=0; j<in_width; j++){
      padded_input->array[padidx+j] = input->array[inidx+j];
    }
  }
  // pad the "bottom", ie the end of the array  
  for (size_t i=pad_top+in_height; i<pad_top + in_height +
	 pad_bottom; i++) {
    size_t idx = i*in_width;
    for (size_t j=0; j<in_width; j++){
      padded_input->array[idx+j] = fill;
    }
  }
}
	       

void k2c_conv1d(k2c_tensor* output, k2c_tensor* input, k2c_tensor* kernel,
		k2c_tensor* bias, size_t stride, size_t dilation,
		   void (*activation) (float[], size_t)) {
  /* 1D (temporal) convolution. Assumes a "channels last" structure
   */

  size_t out_height = output->shape[0];
  size_t out_width = output->shape[1];
  size_t out_size = out_height*out_width;
  size_t in_width = input->shape[1];
  size_t kernel_size = kernel->shape[0];
  
  for (size_t p=0; p < out_height; p++){
    size_t outrowidx = p*out_width;
    for (size_t k=0; k < out_width; k++) {
      for (size_t z=0; z < kernel_size; z++) {
	size_t kernelidx = z*in_width*out_width;
	for (size_t q=0; q < in_width; q++) {
	  size_t inheightidx = q*out_width;
	  output->array[outrowidx+k] +=
	    kernel->array[kernelidx+inheightidx+k]*
	    input->array[(p*stride+z*dilation)*in_width+q];
	}
      }
      output->array[outrowidx+k] += bias->array[k];
    }
  }
  activation(output->array,out_size);
}

#endif /* KERAS2C_CONVOLUTION_LAYERS_H */
