#include <string.h>
#include <stddef.h>
#include <stdarg.h>


void keras2c_add(float output[], size_t numels, size_t num_tensors,...){
/*  Element-wise sum of several tensors. */
/* numels is the number of elements in each tensors */
/* num_tensors is the total number of tensors */
/* results are stored in output array */
/* takes variable number of tensors as inputs: */
/* add(output, numels, num_tensors, tensor1, tensor2, tensor3) etc */
  
  va_list args;
  float *arrptr;
  va_start (args, num_tensors);     
  memset(output, 0, numels*sizeof(*output));
  for (size_t i = 0; i < num_tensors; i++){
    arrptr = va_arg(args, float*);
    for (size_t j=0; j<numels; j++){
      output[j] += arrptr[j];
    }
  }
  va_end (args);             
}

void keras2c_subtract(float output[], size_t numels, size_t num_tensors,
	      float tensor1[], float tensor2[]) {
  /*  Element-wise difference of two tensors. */
  /* output[i] = tensor1[i] - tensor2[i] */
  /* numels is the number of elements in each tensors */
  /* results are stored in output array */

  for (size_t i=0;i<numels;i++){
    output[i] = tensor1[i]-tensor2[i];}
}

void keras2c_multiply(float output[], size_t numels, size_t num_tensors,...){
/*  Element-wise product of several tensors. */
/* numels is the number of elements in each tensors */
/* num_tensors is the total number of tensors */
/* results are stored in output array */
/* takes variable number of tensors as inputs: */
/* multiply(output, numels, num_tensors, tensor1, tensor2,...tensorN) etc */
  
  va_list args;
  float *arrptr;
  va_start (args, num_tensors);

  for (size_t i=0;i<numels;i++){
    output[i] = 1.0f;}
  
  for (size_t i = 0; i < num_tensors; i++){
    arrptr = va_arg(args, float*);
    for (size_t j=0; j<numels; j++){
      output[j] *= arrptr[j];
    }
  }
  va_end (args);             
}

void keras2c_average(float output[], size_t numels, size_t num_tensors,...){
/*  Element-wise average of several tensors. */
/* numels is the number of elements in each tensors */
/* num_tensors is the total number of tensors */
/* results are stored in output array */
/* takes variable number of tensors as inputs: */
/* average(output, numels, num_tensors, tensor1, tensor2,...tensorN) etc */
  
  va_list args;
  float *arrptr;
  float num_tensors_inv = 1.0f/num_tensors;
  
  va_start (args, num_tensors);     
  memset(output, 0, numels*sizeof(*output));
  for (size_t i = 0; i < num_tensors; i++){
    arrptr = va_arg(args, float*);
    for (size_t j=0; j<numels; j++){
      output[j] += arrptr[j]*num_tensors_inv;
    }
  }
  va_end (args);             
}

void keras2c_max(float output[], size_t numels, size_t num_tensors,...){
/*  Element-wise maximum of several tensors. */
/* numels is the number of elements in each tensors */
/* num_tensors is the total number of tensors */
/* results are stored in output array */
/* takes variable number of tensors as inputs: */
/* max(output, numels, num_tensors, tensor1, tensor2,...tensorN) etc */

  
  va_list args;
  float *arrptr;
  va_start (args, num_tensors);
  arrptr = va_arg(args, float*);

  for (size_t i=0;i<numels;i++){
    output[i] = arrptr[i];}
  
  for (size_t i = 0; i < num_tensors-1; i++){
    arrptr = va_arg(args, float*);
    for (size_t j=0; j<numels; j++){
      if (output[j]<arrptr[j]){
	output[j] = arrptr[j];}
    }
  }
  va_end (args);             
}

void keras2c_min(float output[], size_t numels, size_t num_tensors,...){
/*  Element-wise minimum of several tensors. */
/* numels is the number of elements in each tensors */
/* num_tensors is the total number of tensors */
/* results are stored in output array */
/* takes variable number of tensors as inputs: */
/* min(output, numels, num_tensors, tensor1, tensor2,...tensorN) etc */

  va_list args;
  float *arrptr;
  va_start (args, num_tensors);
  arrptr = va_arg(args, float*);

  for (size_t i=0;i<numels;i++){
    output[i] = arrptr[i];}
  
  for (size_t i = 0; i < num_tensors-1; i++){
    arrptr = va_arg(args, float*);
    for (size_t j=0; j<numels; j++){
      if (output[j]>arrptr[j]){
	output[j] = arrptr[j];}
    }
  }
  va_end (args);             
}
