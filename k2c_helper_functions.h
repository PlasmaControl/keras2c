#ifndef KERAS2C_HELPER_FUNCTIONS_H
#define KERAS2C_HELPER_FUNCTIONS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

struct k2c_tensor
{
  float *array;
  size_t shape[4];
  size_t ndim;
};

void keras2c_matmul(float C[], float A[], float B[], size_t outrows,
	    size_t outcols, size_t innerdim) {
  /* Just your basic 1d matrix multiplication. Takes in 1d arrays
 A and B, results get stored in C */
  /*   Size A: outrows*innerdim */
  /*   Size B: innerdim*outcols */
  /*   Size C: outrows*outcols */

  // make sure output is empty
  memset(C, 0, outrows*outcols*sizeof(C[0]));
  
  for (size_t i = 0 ; i < outrows; i++) {
    size_t outrowidx = i*outcols;
    size_t inneridx = i*innerdim;
    for (size_t k = 0; k < innerdim; k++) {
      for (size_t j = 0;  j < outcols; j++) {
	C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
      }
    }
  }
}

void keras2c_affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
	    size_t outcols, size_t innerdim) {
  /* Computes C = A*B + d, where d is a vector that is added to each
 row of A*B*/
  /*   Size A: outrows*innerdim */
  /*   Size B: innerdim*outcols */
  /*   Size C: outrows*outcols */
  /*   Size d: outrows */

  // make sure output is empty
  memset(C, 0, outrows*outcols*sizeof(C[0]));
  
  for (size_t i = 0 ; i < outrows; i++) {
    size_t outrowidx = i*outcols;
    size_t inneridx = i*innerdim;
    for (size_t j = 0;  j < outcols; j++) {
      for (size_t k = 0; k < innerdim; k++) {
	C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
      }
      C[outrowidx+j] += d[j];
    }
  }
}

size_t k2c_sub2idx(size_t sub[], size_t shape[], size_t ndim){
  /* converts from subscript to linear indices in row major order */
  
  size_t idx = 0;
  size_t temp = 0;
  for (size_t i=0; i<ndim; i++) {
    temp = sub[i];
    for (size_t j=ndim; j>i; j--) {
      temp *= shape[j];}
    idx += temp;
    
  }
  return idx;
}

float k2c_vec_dot(float A[], float B[], size_t numels, size_t offsetA,
		  size_t offsetB) {

  float sum = 0.0;
  size_t idxA = 0;
  size_t idxB = 0;
  for (size_t i=0; i<numels; i++) {
    sum += A[idxA]*B[idxB];
    idxA += offsetA;
    idxB += offsetB;
  }
  return sum;
}

void k2c_dot(struct k2c_tensor C, struct k2c_tensor A, struct k2c_tensor B) {

  size_t numelsA = 1;
  size_t numelsB = 1;
  for (size_t i=0; i<A.ndim; i++) {
    numelsA *= A.shape[i];}
  for (size_t i=0; i<B.ndim; i++) {
    numelsB *= B.shape[i];}  
  
  if (A.ndim == 0) {
    for (size_t i=0; i<numelsB; i++) {
      C.array[i] = A.array[0]*B.array[i];}
    return;}
  if (B.ndim == 0) {
    for (size_t i=0; i<numelsA; i++) {
      C.array[i] = A.array[i]*B.array[0];}
    return;}

  for (size_t i=0; i<A.shape[0]; i++) {
    for (size_t j=0; j<A.shape[1]; j++) {
      for (size_t k=0; k<A.shape[2]; k++) {
	for (size_t l=0; l<B.shape[3]; l++) {
	  size_t sub[] = {i,j,k,l};
	  size_t idx = k2c_sub2idx(sub,C.shape,4);
	  size_t offsetA = 



	}
      }
    }
  }
  if (A.ndim == 1) {
    if (B.ndim ==1) {
      C.array[0] = k2c_vec_dot(A.array,B.array,numelsA,1,1);
      return;}
    if (B.ndim==2) {
      for (size_t i=0; i<B.shape[1]; i++) {
	C.array[i] = k2c_vec_dot(A.array,&B.array[i],numelsA,1,B.shape[1]);}
      return;}
    if (B.ndim == 3) {
      

    }
    
    



  }




}



#endif /* KERAS2C_HELPER_FUNCTIONS_H */
