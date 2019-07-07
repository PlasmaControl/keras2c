#ifndef KERAS2C_CONVOLUTION_LAYERS_H
#define KERAS2C_CONVOLUTION_LAYERS_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "k2c_helper_functions.h"

void k2c_pad1d(k2c_tensor* output, k2c_tensor* input, float fill,
	       size_t pad[]) {
  /* pads array in height dimension. 
     fill = fill value
     pad_top: number of rows of fill to concatenate on top of input
     pad_bottom: number of rows of fill to cat on bottom of input */

  size_t in_width = input->shape[1];
  size_t pad_top = pad[0];

  // set output array to fill value
  if (fabs(fill) < 1e-6) {
    // fill is ~zero, use memset
    memset(output->array,0,output->numel*sizeof(output->array[0]));
  }
  else {
    for(size_t i=0; i<output->numel; i++) {
      output->array[i] = fill;}
  }

  // memcpy the old array in the right place
  size_t offset = pad_top*in_width;
  memcpy(&output->array[offset],&input->array[0],
	 input->numel*sizeof(input->array[0]));
}

void k2c_pad2d(k2c_tensor* output, k2c_tensor* input, float fill, size_t pad[]) {

  size_t in_height = input->shape[0];
  size_t in_width = input->shape[1];
  size_t in_channels = input->shape[2];
  size_t pad_top = pad[0];
  size_t pad_left = pad[2];
  size_t pad_right = pad[3];

  // set output array to fill value
  if (fabs(fill) < 1e-6) {
    // fill is ~zero, use memset
    memset(output->array,0,output->numel*sizeof(output->array[0]));
  }
  else {
    for(size_t i=0; i<output->numel; i++) {
      output->array[i] = fill;}
  }
  // memcpy the old array in the middle
  size_t offset = in_channels*(pad_left+pad_right+in_width)*pad_top + in_channels*pad_left;
  size_t num = in_channels*in_width;
  for (size_t i=0; i<in_height; i++) {
    memcpy(&output->array[offset],&input->array[i*num],num*sizeof(input->array[0]));
    offset += num+in_channels*(pad_left+pad_right);
  }
}

void k2c_conv1d(k2c_tensor* output, k2c_tensor* input, k2c_tensor* kernel,
		k2c_tensor* bias, size_t stride, size_t dilation,
		   void (*activation) (float[], size_t)) {
  /* 1D (temporal) convolution. Assumes a "channels last" structure
   */
  memset(output->array,0,output->numel*sizeof(output->array[0]));

  size_t out_times = output->shape[0];
  size_t out_channels = output->shape[1];
  size_t in_channels = input->shape[1];

  for (size_t x0=0; x0 < out_times; x0++){
    for (size_t k=0; k < out_channels; k++) {
      for (size_t z=0; z < kernel->shape[0]; z++) {
	for (size_t q=0; q < in_channels; q++) {
	  size_t outsub[K2C_MAX_NDIM] = {x0,k};
	  size_t inpsub[K2C_MAX_NDIM] = {x0*stride + dilation*z,q};
	  size_t kersub[K2C_MAX_NDIM] = {z,q,k};
	  output->array[k2c_sub2idx(outsub,output->shape,output->ndim)] +=
	    kernel->array[k2c_sub2idx(kersub,kernel->shape,kernel->ndim)]*
	    input->array[k2c_sub2idx(inpsub,input->shape,input->ndim)];
	}
      }
    }
  }
  k2c_bias_add(output,bias);
  activation(output->array,output->numel);
}

void k2c_conv2d(k2c_tensor* output, k2c_tensor* input, k2c_tensor* kernel,
		k2c_tensor* bias, size_t stride[], size_t dilation[],
		   void (*activation) (float[], size_t)) {
  /* 2D (spatial) convolution. Assumes a "channels last" structure
   */
  memset(output->array,0,output->numel*sizeof(output->array[0]));

  size_t out_rows = output->shape[0];
  size_t out_cols = output->shape[1];
  size_t out_channels = output->shape[2];
  size_t in_channels = input->shape[2];

  for (size_t x0=0; x0 < out_rows; x0++){
    for (size_t x1=0; x1 < out_cols; x1++) {
      for (size_t k=0; k < out_channels; k++) {
	for (size_t z0=0; z0 < kernel->shape[0]; z0++) {
	  for (size_t z1=0; z1 < kernel->shape[1]; z1++) {
	    for (size_t q=0; q < in_channels; q++) {
	      size_t outsub[K2C_MAX_NDIM] = {x0,x1,k};
	      size_t inpsub[K2C_MAX_NDIM] = {x0*stride[0] + dilation[0]*z0,
					     x1*stride[1] + dilation[1]*z1,q};
	      size_t kersub[K2C_MAX_NDIM] = {z0,z1,q,k};
	      output->array[k2c_sub2idx(outsub,output->shape,output->ndim)] +=
		kernel->array[k2c_sub2idx(kersub,kernel->shape,kernel->ndim)]*
		input->array[k2c_sub2idx(inpsub,input->shape,input->ndim)];
	    }
	  }
	}
      }
    }
  }
  k2c_bias_add(output,bias);
  activation(output->array,output->numel);
}

void k2c_crop1d(k2c_tensor* output, k2c_tensor* input, size_t crop[]) {

  size_t offset = crop[0]*input->shape[1];
  memcpy(&output->array[0],&input->array[offset],
	 output->numel*sizeof(output->array[0]));
}

void k2c_crop2d(k2c_tensor* output, k2c_tensor* input, size_t crop[]) {

  size_t out_height = output->shape[0];
  size_t in_width = input->shape[1];
  size_t in_channels = input->shape[2];
  size_t crop_top = crop[0];
  size_t crop_left = crop[2];
  size_t crop_right = crop[3];

  size_t offset = in_channels*in_width*crop_top + in_channels*crop_left;
  size_t num = in_channels*(in_width-crop_left-crop_right);
  for (size_t i=0; i<out_height; i++) {
    memcpy(&output->array[i*num],&input->array[offset],num*sizeof(input->array[0]));
    offset += in_width*in_channels;
  }
}

void k2c_upsampling1d(k2c_tensor* output, k2c_tensor* input, size_t size) {

  size_t in_height = input->shape[0];
  size_t in_width = input->shape[1];

  for (size_t i=0; i<in_height; i++) {
    for (size_t j=0; j<size; j++) {
      for (size_t k=0; k<in_width; k++) {
	output->array[(size*i+j)*in_width + k] = input->array[i*in_width+k]; 
      }
    }
  }
}

void k2c_upsampling2d(k2c_tensor* output, k2c_tensor* input, size_t size[]) {

  size_t out_height = output->shape[0];
  size_t out_width = output->shape[1];
  size_t channels = input->shape[2];
  
  for (size_t i=0; i<out_height; i++) {
    for (size_t j=0; j<out_width; j++) {
      size_t insub[K2C_MAX_NDIM] = {i/size[0],j/size[1],0};
      size_t outsub[K2C_MAX_NDIM] = {i,j,0};
      memcpy(&output->array[k2c_sub2idx(outsub,output->shape,output->ndim)],
	     &input->array[k2c_sub2idx(insub,input->shape,input->ndim)],
	     channels*sizeof(input->array[0]));
    }
  }
}
      


#endif /* KERAS2C_CONVOLUTION_LAYERS_H */
