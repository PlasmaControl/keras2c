#ifndef KERAS2C_CORE_LAYERS_H
#define KERAS2C_CORE_LAYERS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include "k2c_helper_functions.h"

void k2c_dense(k2c_tensor* output, k2c_tensor* input, k2c_tensor* kernel,
	       k2c_tensor* bias, void (*activation) (float[], size_t),
	       float fwork[]){

  if (input->ndim <=2) {
    size_t outrows;

    if (input->ndim>1) {
      outrows = input->shape[0];}
    else {
      outrows = 1;}
    size_t outcols = kernel->shape[1];
    size_t innerdim = kernel->shape[0];
    size_t outsize = outrows*outcols;
    k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
		      outrows,outcols,innerdim);
    activation(output->array,outsize);
  }
  else {
    size_t axesA[1] = {input->ndim-1};
    size_t axesB[1] = {0};
    size_t naxes = 1;
    int normalize = 0;

    k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
  }
}

void k2c_flatten(k2c_tensor* input) {

  for (size_t i=0; i<input->ndim; i++) {
    input->shape[i] = 1;}
  input->shape[0] = input->numel;
  input->ndim = 1;
}

void k2c_reshape(k2c_tensor* input, size_t newshp[], size_t newndim) {
  for (size_t i=0; i<input->ndim; i++) {
    input->shape[i] = 1;}
  for (size_t i=0; i<newndim; i++) {
    input->shape[i] = newshp[i];}
  input->ndim = newndim;
}

void k2c_permute_dims(k2c_tensor* output, k2c_tensor* input, 
		      size_t permute[]) {

  size_t Asub[K2C_MAX_NDIM];
  size_t Bsub[K2C_MAX_NDIM];
  size_t newshp[K2C_MAX_NDIM];
  size_t oldshp[K2C_MAX_NDIM];
  size_t ndim = input->ndim;
  size_t bidx=0;
  for (size_t i=0; i<ndim; i++) {
    oldshp[i] = input->shape[i];}  
  for (size_t i=0; i<ndim; i++) {
    newshp[i] = oldshp[permute[i]];}
  
  for (size_t i=0; i<input->numel; i++) {
    k2c_idx2sub(i,Asub,oldshp,ndim);
    for (size_t j=0; j<ndim; j++) {
      Bsub[j] = Asub[permute[j]];}
    bidx = k2c_sub2idx(Bsub,newshp,ndim);
    output->array[bidx] = input->array[i];
  }
}

void k2c_repeat_vector(k2c_tensor* output, k2c_tensor* input, size_t n) {

  size_t in_width = input->shape[0];
  for (size_t i=0; i<n; i++) {
    for(size_t j=0; j<in_width; j++) {
      output->array[i*in_width + j] = input->array[j];}
  }
}



#endif /* KERAS2C_CORE_LAYERS_H */
