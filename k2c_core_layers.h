#ifndef KERAS2C_CORE_LAYERS_H
#define KERAS2C_CORE_LAYERS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include "keras2c_helper_functions.h"

void keras2c_dense(float output[], float input[], float kernel[], float bias[],
	   size_t outrows, size_t outcols, size_t innerdim,
	   void (*activation) (float[], size_t)){
  //  printf("in dense \n");
  size_t outsize = outrows*outcols;
  keras2c_affine_matmul(output,input,kernel,bias,outrows,outcols,innerdim);
  activation(output,outsize);
}

void keras2c_permute_dims(float output[], float input[], size_t ndim,
			  size_t olddim[], size_t permute[]) {

  size_t oldidx[5];
  size_t newidx[5];
  size_t newdim[5];

  if (ndim==1){
    //do nothing
     // no need to transpose a 1d array
  }
  if (ndim==2) {
    newdim[0] = olddim[permute[0]];
    newdim[1] = olddim[permute[1]];
    for (size_t i=0; i<olddim[0]; i++) {
      for (size_t j=0; j<olddim[1]; j++) {
	oldidx[0] = i;
	oldidx[1] = j;
	newidx[0] = oldidx[permute[0]];
	newidx[1] = oldidx[permute[1]];
	output[newidx[0]*newdim[1] +
	       newidx[1]]
	  = input[oldidx[0]*olddim[1] +
		  oldidx[1]];
      }
    }
  }
  if (ndim==3) {
    newdim[0] = olddim[permute[0]];
    newdim[1] = olddim[permute[1]];
    newdim[2] = olddim[permute[2]];
    for (size_t i=0; i<olddim[0]; i++) {
      for (size_t j=0; j<olddim[1]; j++) {
	for (size_t k=0; k<olddim[2]; k++) {
	oldidx[0] = i;
	oldidx[1] = j;
	oldidx[2] = k;
	newidx[0] = oldidx[permute[0]];
	newidx[1] = oldidx[permute[1]];
	newidx[2] = oldidx[permute[2]];
        output[newidx[0]*newdim[1]*newdim[2] +
	       newidx[1]*newdim[2] +
	       newidx[2]]
	  = input[oldidx[0]*olddim[1]*olddim[2] +
		  oldidx[1]*olddim[2] +
		  oldidx[2]];
	}
      }
    }
  }
  if (ndim==4) {
    newdim[0] = olddim[permute[0]];
    newdim[1] = olddim[permute[1]];
    newdim[2] = olddim[permute[2]];
    newdim[3] = olddim[permute[3]];
    for (size_t i=0; i<olddim[0]; i++) {
      for (size_t j=0; j<olddim[1]; j++) {
	for (size_t k=0; k<olddim[2]; k++) {
	  for (size_t l=0; l<olddim[3]; l++) {
	oldidx[0] = i;
	oldidx[1] = j;
	oldidx[2] = k;
	oldidx[3] = l;
	newidx[0] = oldidx[permute[0]];
	newidx[1] = oldidx[permute[1]];
	newidx[2] = oldidx[permute[2]];
	newidx[3] = oldidx[permute[3]];
        output[newidx[0]*newdim[1]*newdim[2]*newdim[3] +
	       newidx[1]*newdim[2]*newdim[3] +
	       newidx[2]*newdim[3] +
	       newidx[3]]
	  = input[oldidx[0]*olddim[1]*olddim[2]*olddim[3] +
		  oldidx[1]*olddim[2]*olddim[3] +
		  oldidx[2]*olddim[3] +
		  oldidx[3]];
	  }
	}
      }
    }
  }
}
  



#endif /* KERAS2C_CORE_LAYERS_H */
