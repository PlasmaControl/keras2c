#ifndef KERAS2C_HELPER_FUNCTIONS_H
#define KERAS2C_HELPER_FUNCTIONS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define K2C_MAX_NDIM 5

struct k2c_tensor
{
  float *array;
  size_t ndim;
  size_t numel;
  size_t shape[K2C_MAX_NDIM];
};
typedef struct k2c_tensor k2c_tensor;


void k2c_matmul(float C[], float A[], float B[], size_t outrows,
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

void k2c_affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
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

void k2c_sub2idx(size_t idx, size_t sub[], size_t shape[], size_t ndim){
  /* converts from subscript to linear indices in row major order */

  idx = 0;
  size_t temp = 0;
  for (size_t i=0; i<ndim; i++) {
    temp = sub[i];
    for (size_t j=ndim; j>i; j--) {
      temp *= shape[j];}
    idx += temp;
    
  }
}

void k2c_idx2sub(size_t idx, size_t sub[], size_t shape[], size_t ndim) {

  
  for (int i=ndim-1; i>=0; i--) {
    sub[i] = idx%shape[i];
    idx /= shape[i];
  }


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

void k2c_dot(k2c_tensor C, k2c_tensor A, k2c_tensor B, size_t axesA[],
	     size_t axesB[], size_t naxes, int normalize, float fwork[]) {

  size_t permA[K2C_MAX_NDIM];
  size_t permB[K2C_MAX_NDIM];
  size_t prod_axesA = 1;
  size_t prod_axesB = 1;
  size_t free_axesA, free_axesB;
  size_t freeA[K2C_MAX_NDIM];
  size_t freeB[K2C_MAX_NDIM];
  size_t count=0;
  int isin;
  size_t i,j;
  size_t newshpA[K2C_MAX_NDIM];
  size_t newshpB[K2C_MAX_NDIM];
  size_t ndimA = A.ndim;
  size_t ndimB = B.ndim;
  float *reshapeA, *reshapeB;

  // find which axes are free (ie, not being summed over)
  for (i=0; i<A.ndim; i++) {
    isin = 0;
    for (j=0; j<naxes; j++) {
      if (i==axesA[j]) {
	isin=1;}
    }
    if (!isin) {
      freeA[count] = i;
      count++;
    }
  }
  for (i=0; i<ndimB; i++) {
    isin = 0;
    for (j=0; j<naxes; j++) {
      if (i==axesB[j]) {
	isin=1;}
    }
    if (!isin) {
      freeB[count] = i;
      count++;
    }
  }

  // temp working storage
  reshapeA = &fwork[0];
  reshapeB = &fwork[A.numel];
    // number of elements in inner dimension
  for (i=0; i < naxes; i++) {
    prod_axesA *= A.shape[axesA[i]];}
  for (i=0; i < naxes; i++) {
    prod_axesB *= B.shape[axesB[i]];}
  // number of elements in free dimension
  free_axesA = A.numel/prod_axesA;
  free_axesB = B.numel/prod_axesB;
  // find permutation of axes to get into matmul shape
  for (i=0; i<ndimA-naxes; i++) {
    permA[i] = freeA[i];}
  for (i=ndimA-naxes, j=0; i<ndimA; i++, j++) {
    permA[i] = axesA[j];}
  for (i=0; i<naxes; i++) {
    permB[i] = axesB[i];}
  for (i=naxes, j=0; i<ndimB; i++, j++) {
    permB[i] = freeB[j];}

  size_t Asub[K2C_MAX_NDIM];
  size_t Bsub[K2C_MAX_NDIM];
  size_t bidx=0;
  for (i=0; i<A.ndim; i++) {
    newshpA[i] = A.shape[permA[i]];
  }
  for (i=0; i<B.ndim; i++) {
    newshpB[i] = B.shape[permB[i]];
  }

  // reshape arrays
  for (i=0; i<A.numel; i++) {
    k2c_idx2sub(i,Asub,A.shape,ndimA);
    for (j=0; j<ndimA; j++) {
      Bsub[j] = Asub[permA[j]];}
    k2c_sub2idx(bidx,Bsub,newshpA,ndimA);
    reshapeA[bidx] = A.array[i];
  }

  for (i=0; i<B.numel; i++) {
    k2c_idx2sub(i,Asub,B.shape,ndimA);
    for (j=0; j<ndimB; j++) {
      Bsub[j] = Asub[permB[j]];}
    k2c_sub2idx(bidx,Bsub,newshpA,ndimA);
    reshapeB[bidx] = B.array[i];
  }
  if (normalize) { 
    float sum;
    float inorm;
    for (size_t i=0; i<free_axesA; i++) {
      sum = 0;
      for (size_t j=0; j<prod_axesA; j++) {
	sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];}
      inorm = 1.0f/sqrt(sum);
      for (size_t j=0; j<prod_axesA; j++) {
	reshapeA[i*prod_axesA + j] *= inorm;}
    }
    for (size_t i=0; i<free_axesB; i++) {
      sum = 0;
      for (size_t j=0; j<prod_axesB; j++) {
	sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];}
      inorm = 1.0f/sqrt(sum);
      for (size_t j=0; j<prod_axesB; j++) {
	reshapeB[i + free_axesB*j] *= inorm;}
    }
  }
  k2c_matmul(C.array, reshapeA, reshapeB, free_axesA,
	     free_axesB, prod_axesA);
}

void k2c_bias_add(k2c_tensor A, k2c_tensor b) {
  /* adds bias vector b to tensor A. Assumes b is a rank 1 tensor */
  /* that is added to the last dimension of A */
  for (size_t i=0; i<A.numel; i+=b.numel) {
    for (size_t j=0; j<b.numel; j++) {
      A.array[i+j] += b.array[j];}
  }
}


#endif /* KERAS2C_HELPER_FUNCTIONS_H */
