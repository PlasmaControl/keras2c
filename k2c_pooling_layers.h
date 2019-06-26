#ifndef KERAS2C_POOLING_LAYERS_H
#define KERAS2C_POOLING_LAYERS_H

#include <math.h>
#include <stddef.h>
#include <string.h>
#include "k2c_helper_functions.h"

void k2c_global_max_pooling_1d(k2c_tensor *output, k2c_tensor *input) {

  size_t in_height = input->shape[0];
  size_t in_width = input->shape[1];
  
  for (size_t i=0; i<in_width; i++){
    output->array[i] = input->array[i];}

  for (size_t j=1; j<in_height; j++){
    size_t rowidx = j*in_width;
    for (size_t i=0; i<in_width;i++){
      if (output->array[i]<input->array[rowidx+i]){
	output->array[i] = input->array[rowidx+i];
      }
    }
  }
}

void k2c_global_avg_pooling_1d(k2c_tensor *output, k2c_tensor *input) {

  size_t in_height = input->shape[0];
  size_t in_width = input->shape[1];
  memset(output,0,in_width*sizeof(*input));
  float in_height_inv = 1.0f/in_height;
  
  for (size_t j=0; j<in_height; j++){
    size_t rowidx = j*in_width;
    for (size_t i=0; i<in_width;i++){
      output->array[i] += input->array[rowidx+i]*in_height_inv;
    }
  }
}

float k2c_arrmax(float array[], size_t numels, size_t offset){

  float max=array[0];
  for (size_t i=1; i<numels; i++){
    if (array[i*offset]>max){
      max = array[i*offset];}
  }
  return max;
}

float k2c_arravg(float array[], size_t numels, size_t offset){

  float avg=0.0f;
  size_t count = 0;
  
  for (size_t i=0; i<numels; i++){
    if (array[i*offset] > -HUGE_VALF) {
      avg  += array[i*offset];
      count += 1;
    }
  }
  avg = avg/count;
  return avg;
}

void k2c_maxpool1d(k2c_tensor *output, k2c_tensor *input, size_t pool_size,
	       size_t stride){

  size_t in_width = input->shape[1];
  size_t out_height = output->shape[0];
  
  for (size_t i=0, k=0; i<out_height; i++, k+=stride){
    size_t inrowidx = k*in_width;
    size_t outrowidx = i*in_width;
    for (size_t j=0; j<in_width; j++){
      output->array[outrowidx+j] = k2c_arrmax(&input->array[inrowidx+j],
				   pool_size,in_width);
    }
  }
}

void k2c_avgpool1d(k2c_tensor *output, k2c_tensor *input, size_t pool_size,
	       size_t stride){

    size_t in_width = input->shape[1];
    size_t out_height = output->shape[0];

    for (size_t i=0, k=0; i<out_height; i++, k+=stride){
      size_t inrowidx = k*in_width;
      size_t outrowidx = i*in_width;
      for (size_t j=0; j<in_width; j++){
	output->array[outrowidx+j] = k2c_arravg(&input->array[inrowidx+j],
						pool_size,in_width);
    }
  }
}



#endif /* KERAS2C_POOLING_LAYERS */

