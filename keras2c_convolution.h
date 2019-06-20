#ifndef KERAS2C_CONVOLUTIONS_H
#define KERAS2C_CONVOLUTIONS_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>


void keras2c_pad1d(float input[], float padded_input[], float fill[],
	       size_t in_height, size_t in_width, size_t pad_top,
	       size_t pad_bottom) {
  /* pads array in height dimension. 
     fill = 1xW array, where W is the width of the input array
     pad_top: number of copies of fill to concatenate on top of input
     pad_bottom: number of copies of fill to cat on bottom of input */
  // printf("in padding1d \n");

  // pad the "top", ie, beginning of array
  for (size_t i=0; i<pad_top; i++) {
    size_t idx = i*in_width;
    for (size_t j=0; j<in_width; j++){
      padded_input[idx+j] = fill[j];
    }
  }
  // put the original values in the middle
  for (size_t i=pad_top, k=0; i<pad_top+in_height; i++, k++){
    size_t padidx = i*in_width;
    size_t inidx = k*in_width;
    for (size_t j=0; j<in_width; j++){
      padded_input[padidx+j] = input[inidx+j];
    }
  }
  // pad the "bottom", ie the end of the array  
  for (size_t i=pad_top+in_height; i<pad_top + in_height +
	 pad_bottom; i++) {
    size_t idx = i*in_width;
    for (size_t j=0; j<in_width; j++){
      padded_input[idx+j] = fill[j];
    }
  }
}
	       



void keras2c_conv1d(float input[], float output[], float kernel[],
		   float bias[], size_t out_height, size_t out_width,
		   size_t kernel_size, size_t padded_in_height,
		   size_t in_width, size_t stride, size_t dilation,
		   void (*activation) (float[], size_t)) {
  /* 1D (temporal) convolution. Assumes a "channels last" structure
   */

  /*  idx = (3,2,5) */
  /* b[idx] - a[idx[0] * (in_width * out_width) + (idx[1] * in_width) + idx[2]] */
    
  // printf("in conv1d \n");
  int out_size = out_height*out_width;
  for (size_t p=0; p < out_height; p++){
    size_t outrowidx = p*out_width;
    for (size_t k=0; k < out_width; k++) {
      for (size_t z=0; z < kernel_size; z++) {
	size_t kernelidx = z*in_width*out_width;
	for (size_t q=0; q < in_width; q++) {
	  size_t inheightidx = q*out_width;
	  output[outrowidx+k] +=
	    kernel[kernelidx+inheightidx+k]*
	    input[(p*stride+z*dilation)*in_width+q];
	}
      }
      output[outrowidx+k] += bias[k];
    }
  }
  activation(output,out_size);
}

#endif /* KERAS2C_CONVOLUTIONS_H */
