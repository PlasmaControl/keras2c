#ifndef KERAS2C_HELPER_FUNCTIONS_H
#define KERAS2C_HELPER_FUNCTIONS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

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


#endif /* KERAS2C_HELPER_FUNCTIONS_H */
