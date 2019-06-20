#ifndef KERAS2C_RECURRENT_H
#define KERAS2C_RECURRENT_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "keras2c_helper_functions.h"

void keras2c_lstmcell(float input[], float state[], float kernel[],
		      float recurrent_kernel[], float bias[], size_t units,
		      size_t in_height, size_t in_width, float fwork[],
		      void (*recurrent_activation) (float*, size_t),
		      void (*output_activation)(float*, size_t)){


  float *h_tm1 = &state[0];  // previous memory state
  float *c_tm1 = &state[units];  // previous carry state
  size_t outrows = 1;
  float *Wi = &kernel[0];
  float *Wf = &kernel[in_width*units];
  float *Wc = &kernel[2*in_width*units];
  float *Wo = &kernel[3*in_width*units];
  float *Ui = &recurrent_kernel[0];
  float *Uf = &recurrent_kernel[units*units];
  float *Uc = &recurrent_kernel[2*units*units];
  float *Uo = &recurrent_kernel[3*units*units];
  float *bi = &bias[0];
  float *bf = &bias[units];
  float *bc = &bias[2*units];
  float *bo = &bias[3*units];
  float *xi = &fwork[0];
  float *xf = &fwork[units];
  float *xc = &fwork[2*units];
  float *xo = &fwork[3*units];
  float *yi = &fwork[4*units];
  float *yf = &fwork[5*units];
  float *yc = &fwork[6*units];
  float *yo = &fwork[7*units];
 

  //xi = input*Wi + bi;
  keras2c_affine_matmul(xi, input, Wi, bi, outrows, units, in_width);
  //xf = input*Wf + bf;
  keras2c_affine_matmul(xf, input, Wf, bf, outrows, units, in_width);
  //xc = input*Wc + bc;
  keras2c_affine_matmul(xc, input, Wc, bc, outrows, units, in_width);
  //xo = input*Wo + bo;
  keras2c_affine_matmul(xo, input, Wo, bo, outrows, units, in_width);
 
  // yi = recurrent_activation(xi + h_tm1*Ui);
  keras2c_affine_matmul(yi, h_tm1, Ui, xi, outrows, units, units);
  recurrent_activation(yi, units);

  // yf = recurrent_activation(xf + h_tm1*Uf); 
  keras2c_affine_matmul(yf, h_tm1, Uf, xf, outrows, units, units);
  recurrent_activation(yf, units);

  // yc = yf.*c_tm1 + yi.*output_activation(xc + h_tm1*Uc);
  keras2c_affine_matmul(yc, h_tm1, Uc, xc, outrows, units, units);
  output_activation(yc, units);
  for (size_t i=0; i < units; i++){
    yc[i] = yf[i]*c_tm1[i] + yi[i]*yc[i];}

  
  // yo = recurrent_activation(xo + h_tm1*Uo); 
  keras2c_affine_matmul(yo, h_tm1, Uo, xo, outrows, units, units);
  recurrent_activation(yo, units);

  // h = yo.*output_activation(yc); 
  // state = [h;yc];
  for (size_t i=0; i < units; i++){
    state[units+i] = yc[i];}

  output_activation(yc, units);

  for (size_t i=0; i < units; i++){
    state[i] = yo[i]*yc[i];}

}

void keras2c_lstm(float input[], float state[], float kernel[],
		  float recurrent_kernel[], float bias[], size_t units,
		  size_t in_height, size_t in_width, float fwork[],
		  void (*recurrent_activation) (float*, size_t),
		  void (*output_activation)(float*, size_t), float *output){


    
    
  for (size_t i=0; i < in_height; i++){
    keras2c_lstmcell(&input[i*in_width], state, kernel, recurrent_kernel,
		     bias, units, in_height, in_width, fwork,
		     recurrent_activation, output_activation);
  }
  for (size_t i=0; i < units; i++){
    output[i] = state[i];
  }
}


void keras2c_simpleRNNcell(float input[], float state[], float kernel[],
			   float recurrent_kernel[], float bias[], size_t units,
			   size_t in_height, size_t in_width, float fwork[],
			   void (*output_activation)(float*, size_t)) {
      printf("start rnn_cell, input: \n");
  for (int i=0; i<in_width;i++){
    printf("%.2f  ", input[i]);}
  printf("\n");
        printf("start rnn_cell, state: \n");
  for (int i=0; i<units;i++){
    printf("%.2f  ", state[i]);}
  printf("\n");
  
  size_t outrows = 1;
  float *h1 = &fwork[0];
  float *h2 = &fwork[units];
  // h1 = input*kernel+bias
  keras2c_affine_matmul(h1,input,kernel,bias,outrows,units,in_width);
        printf("after first matmul, h1=: \n");
  for (int i=0; i<units;i++){
    printf("%.2f  ", h1[i]);}
  printf("\n");
  // h2 = state*recurrent_kernel + h1
  keras2c_affine_matmul(h2,state,recurrent_kernel,h1,outrows,units,units);
          printf("after second matmul, h2=: \n");
  for (int i=0; i<units;i++){
    printf("%.2f  ", h2[i]);}
  printf("\n");
  output_activation(h2,units);
          printf("after activation, h2=: \n");
  for (int i=0; i<units;i++){
    printf("%.2f  ", h2[i]);}
  printf("\n");
  for (size_t i=0;i<units;i++) {
    state[i] = h2[i];
  }

          printf("end rnn_cell, state: \n");
  for (int i=0; i<units;i++){
    printf("%.2f  ", state[i]);}
  printf("\n");
}

void keras2c_simpleRNN(float input[], float state[], float kernel[],
		       float recurrent_kernel[], float bias[], size_t units,
		       size_t in_height, size_t in_width, float fwork[],
		       void (*output_activation)(float*, size_t), float *output) {
  for (size_t i=0; i<in_height; i++) {
    keras2c_simpleRNNcell(&input[i*in_width],state,kernel,recurrent_kernel,bias,
			  units,in_height,in_width,fwork, output_activation);
  }
  for (size_t i=0; i < units; i++){
    output[i] = state[i];
  }
}


void keras2c_grucell(float input[], float state[], float kernel[],
		     float recurrent_kernel[], float bias[], size_t units,
		     size_t in_height, size_t in_width, float fwork[],
		     void (*recurrent_activation)(float*, size_t),
		     void (*output_activation)(float*, size_t), int reset_after) {

  float *h_tm1 = &state[0];
  size_t outrows = 1;
  float *Wz = &kernel[0];
  float *Wr = &kernel[in_width*units];
  float *Wh = &kernel[2*in_width*units];
  float *Uz = &recurrent_kernel[0];
  float *Ur = &recurrent_kernel[units*units];
  float *Uh = &recurrent_kernel[2*units*units];
  float *bz = &bias[0];
  float *br = &bias[units];
  float *bh = &bias[2*units];
  float *rbz = &bias[3*units];
  float *rbr = &bias[4*units];
  float *rbh = &bias[5*units];
  float *xz = &fwork[0];
  float *xr = &fwork[units];
  float *xh = &fwork[2*units];
  float *yz = &fwork[3*units];
  float *yr = &fwork[4*units];
  float *yh = &fwork[5*units];

  //     x_z = input*kernel_z + input_bias_z
  keras2c_affine_matmul(xz, input, Wz, bz, outrows, units, in_width);
  //    x_r = input@kernel_r + input_bias_r
  keras2c_affine_matmul(xr, input, Wr, br, outrows, units, in_width);
  //    x_h = input@kernel_h + input_bias_h
  keras2c_affine_matmul(xh, input, Wh, bh, outrows, units, in_width);

  //   recurrent_z = h_tm1@recurrent_kernel_z
  keras2c_affine_matmul(yz, h_tm1, Uz, rbz, outrows, units, units);
  //    recurrent_r = h_tm1@recurrent_kernel_r
  keras2c_affine_matmul(yr, h_tm1, Ur, rbr, outrows, units, units);

  //    z = np.tanh(x_z + recurrent_z)
  //    r = np.tanh(x_r + recurrent_r)
  for (size_t i=0; i<units; i++) {
    yz[i] = xz[i] + yz[i];
    yr[i] = xr[i] + yr[i];
  }
  recurrent_activation(yz, units);
  recurrent_activation(yr, units);

  //    reset gate applied after/before matrix multiplication
  if (reset_after) {
  //        recurrent_h = h_tm1@recurrent_kernel_h + recurrent_bias_h
    keras2c_affine_matmul(yh, h_tm1, Uh, rbh, outrows, units, units);
    //        recurrent_h = r .* recurrent_h
    for (size_t i=0; i<units; i++) {
      yh[i] = yr[i] * yh[i];}
  }
  else {
    //        recurrent_h = (r .* h_tm1)@recurrent_kernel_h
    for (size_t i=0; i<units; i++) {
      yh[i] = yr[i]*h_tm1[i];}
    keras2c_matmul(xz, yh, Uh, outrows, units, units); //reuse xz as new yh
  }
  //    hh = np.tanh(x_h + recurrent_h)
  for (size_t i=0; i<units; i++) {
    xr[i] = xh[i] + xz[i];} // reuse xr = hh
  output_activation(xr, units);

  //    h = z .* h_tm1 + (1 - z) .* hh
  for (size_t i=0; i<units; i++) {
    state[i] = yz[i] * h_tm1[i] + (1.0f-yz[i])*xr[i];}
}


void keras2c_gru(float input[], float state[], float kernel[],
		 float recurrent_kernel[], float bias[], size_t units,
		 size_t in_height, size_t in_width, float fwork[],
		 void (*recurrent_activation)(float*, size_t),
		 void (*output_activation)(float*, size_t), int reset_after,
		 float output[]) {

  for (size_t i=0; i<in_height; i++) {
    keras2c_grucell(&input[i*in_width], state, kernel, recurrent_kernel, bias,
		    units, in_height, in_width, fwork, recurrent_activation,
		    output_activation, reset_after);
  }
  for (size_t i=0; i<units; i++) {
    output[i] = state[i];
  }
}




#endif /* KERAS2C_RECURRENT_H */
