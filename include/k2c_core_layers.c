#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "k2c_include.h"

void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
	       const k2c_tensor* bias, k2c_activationType *activation, float fwork[]){

  if (input->ndim <=2) {
    size_t outrows;

    if (input->ndim>1) {
      outrows = input->shape[0];}
    else {
      outrows = 1;}
    const size_t outcols = kernel->shape[1];
    const size_t innerdim = kernel->shape[0];
    const size_t outsize = outrows*outcols;
    k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
		      outrows,outcols,innerdim);
    activation(output->array,outsize);
  }
  else {
    const size_t axesA[1] = {input->ndim-1};
    const size_t axesB[1] = {0};
    const size_t naxes = 1;
    const int normalize = 0;

    k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
  }
}


void k2c_flatten(k2c_tensor *output, const k2c_tensor* input) {

  memcpy(output->array, input->array, input->numel*sizeof(input->array[0]));
  for (size_t i=0; i<input->ndim; i++) {
    output->shape[i] = 1;}
  output->shape[0] = input->numel;
  output->numel = input->numel;
  output->ndim = 1;
}


void k2c_reshape(k2c_tensor *output, const k2c_tensor* input, const size_t newshp[],
		 const size_t newndim) {
  
  memcpy(output->array, input->array, input->numel*sizeof(input->array[0]));
  for (size_t i=0; i<newndim; i++) {
    output->shape[i] = newshp[i];}
  output->ndim = newndim;
  output->numel = input->numel;
}


void k2c_permute_dims(k2c_tensor* output, const k2c_tensor* input, 
		      const size_t permute[]) {

  size_t Asub[K2C_MAX_NDIM];
  size_t Bsub[K2C_MAX_NDIM];
  size_t newshp[K2C_MAX_NDIM];
  size_t oldshp[K2C_MAX_NDIM];
  const size_t ndim = input->ndim;
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

void k2c_repeat_vector(k2c_tensor* output, const k2c_tensor* input, const size_t n) {

  const size_t in_width = input->shape[0];
  for (size_t i=0; i<n; i++) {
    for(size_t j=0; j<in_width; j++) {
      output->array[i*in_width + j] = input->array[j];}
  }
}
