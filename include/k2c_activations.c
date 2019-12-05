#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "k2c_include.h"


/* typedef void k2c_activationType(float x[], const size_t size); */


// Regular Activations
//*****************************************************************************
void k2c_linear_func(float x[], const size_t size){
  /* linear activation. Doesn't do anything, just a dummy fn */

}
k2c_activationType * k2c_linear = k2c_linear_func;

void k2c_exponential_func(float x[], const size_t size){
  /* exponential activation */
  /* y = exp(x) */
  /* x is overwritten with the activated values */

  for (size_t i=0; i<size; i++) {
    x[i] = exp(x[i]);}
}
k2c_activationType * k2c_exponential = k2c_exponential_func;

void k2c_relu_func(float x[], const size_t size) {
  /* Rectified Linear Unit activation (ReLU) */
  /*   y = max(x,0) */
  /* x is overwritten with the activated values */

  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = 0.0f;
    }
  }
}
k2c_activationType * k2c_relu = k2c_relu_func;

void k2c_hard_sigmoid_func(float x[], const size_t size) {
  /* Hard Sigmoid activation */
  /*   y = {1 if x> 2.5 */
  /*        0.2*x+0.5 if -2.5<x<2.5 */
  /*        0 if x<2.5 */
  /* x is overwritten with the activated values */

  for (size_t i=0; i < size; i++) {
    if (x[i] <= -2.5f){
      x[i] = 0.0f;
    }
    else if (x[i]>=2.5f) {
      x[i] = 1.0f;
    }
    else {
      x[i] = 0.2f*x[i] + 0.5f;
    }
  }
}
k2c_activationType * k2c_hard_sigmoid = k2c_hard_sigmoid_func;


void k2c_tanh_func(float x[], const size_t size) {
  /* standard tanh activation */
  /* x is overwritten with the activated values */
  for (size_t i=0; i<size; i++){
    x[i] = tanh(x[i]);
  }
}
k2c_activationType * k2c_tanh = k2c_tanh_func;


void k2c_sigmoid_func(float x[], const size_t size) {
  /* Sigmoid activation */
  /*   y = 1/(1+exp(-x)) */
  /* x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = 1/(1+exp(-x[i]));
  }
}
k2c_activationType * k2c_sigmoid = k2c_sigmoid_func;

void k2c_softmax_func(float x[], const size_t size) {
  /* Softmax activation */
  /*     z[i] = exp(x[i]-max(x)) */
  /*     y = z/sum(z) */
  /* x is overwritten with the activated values */

  float xmax = x[0];
  float sum = 0;
  for (size_t i=0; i < size; i++) {
    if (x[i]>xmax) {
      xmax = x[i];}
  }

  for (size_t i=0; i < size; i++) {
    x[i] = exp(x[i]-xmax);
  }

  for (size_t i=0; i < size; i++) {
    sum += x[i];
  }

  sum = 1.0f/sum; // divide once and multiply -> fast
  for (size_t i=0; i < size; i++) {
    x[i] = x[i]*sum;
  }
}
k2c_activationType * k2c_softmax = k2c_softmax_func;

void k2c_softplus_func(float x[], const size_t size) {
  /* Softplus activation */
  /*   y = ln(1+exp(x)) */
  /*   x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = log1p(exp(x[i]));
  }
}
k2c_activationType * k2c_softplus = k2c_softplus_func;

void k2c_softsign_func(float x[], const size_t size) {
  /* Softsign activation */
  /*   y = x/(1+|x|) */
  /*   x is overwritten by the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = x[i]/(1.0f + fabs(x[i]));
  }
}
k2c_activationType * k2c_softsign = k2c_softsign_func;

// Advanced Activations
//*****************************************************************************

void k2c_LeakyReLU(float x[], const size_t size, const float alpha){
  /* Leaky version of a Rectified Linear Unit. */
  /* It allows a small gradient when the unit is not active: */
  /*   f(x) = alpha * x for x < 0, f(x) = x for x >= 0. */
  /*   x is overwritten by the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i]<0){
      x[i] = alpha*x[i];}
  }
}

void k2c_PReLU(float x[], const size_t size, const float alpha[]) {
  /*  Parametric Rectified Linear Unit. */
  /*  f(x) = alpha * x for x < 0, f(x) = x for x >= 0, */
  /*  where alpha is a learned array with the same shape as x. */
  /*  x is overwritten by the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i]<0.0f) {
      x[i] = x[i]*alpha[i];}
  }
}
  

void k2c_ELU(float x[], const size_t size, const float alpha) {
  /* Exponential Linear Unit activation (ELU) */
  /*   y = {x if x>0 */
  /* 	 alpha*(e^x - 1) else} */
  /* x is overwritten with the activated values */
    
  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = alpha*expm1(x[i]);
    }
  }
}

void k2c_ThresholdedReLU(float x[], const size_t size, const float theta) {
  /* Thresholded Rectified Linear Unit. */
  /*   f(x) = x for x > theta, f(x) = 0 otherwise. */
  /* x is overwritten with the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i]<= theta) {
      x[i] = 0;}
  }
}


void k2c_ReLU(float x[], const size_t size, const float max_value,
	      const float negative_slope, const float threshold) {
  /* Rectified Linear Unit activation function. */
  /* f(x) = max_value for x >= max_value, */
  /* f(x) = x for threshold <= x < max_value, */
  /* f(x) = negative_slope * (x - threshold) otherwise. */
  /* x is overwritten with the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i] >= max_value) {
      x[i] = max_value;}
    else if (x[i] < threshold) {
      x[i] = negative_slope*(x[i] - threshold);}
  }
}
