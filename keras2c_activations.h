#ifndef KERAS2C_ACTIVATIONS_H
#define KERAS2C_ACTIVATIONS_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>


// Regular Activations
//*****************************************************************************
void keras2c_linear(float x[], size_t size){
  /* linear activation. Doesn't do anything, just a dummy fn */

}

void keras2c_exponential(float x[], size_t size){
  /* exponential activation */
  /* y = exp(x) */
  /* x is overwritten with the activated values */

  for (size_t i; i<size; i++) {
    x[i] = exp(x[i]);}
}

void keras2c_relu(float x[], size_t size) {
  /* Rectified Linear Unit activation (ReLU) */
  /*   y = max(x,0) */
  /* x is overwritten with the activated values */

  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = 0.0f;
    }
  }
}

void keras2c_hard_sigmoid(float x[], size_t size) {
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

void keras2c_tanh(float x[], size_t size) {
  /* standard tanh activation */
  /* x is overwritten with the activated values */
  for (size_t i=0; i<size; i++){
    x[i] = tanh(x[i]);
  }
}

void keras2c_sigmoid(float x[], size_t size) {
  /* Sigmoid activation */
  /*   y = 1/(1+exp(-x)) */
  /* x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = 1/(1+exp(-x[i]));
  }
}

void keras2c_softmax(float *x, size_t size) {
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

void keras2c_softplus(float x[], size_t size) {
  /* Softplus activation */
  /*   y = ln(1+exp(x)) */
  /*   x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = log(1.0f + exp(x[i]));
  }
}

void keras2c_softsign(float x[], size_t size) {
  /* Softsign activation */
  /*   y = x/(1+|x|) */
  /*   x is overwritten by the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = x[i]/(1.0f + fabs(x[i]));
  }
}

// Advanced Activations
//*****************************************************************************

void keras2c_LeakyReLU(float x[], size_t size, float alpha){
  /* Leaky version of a Rectified Linear Unit. */
  /* It allows a small gradient when the unit is not active: */
  /*   f(x) = alpha * x for x < 0, f(x) = x for x >= 0. */
  /*   x is overwritten by the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i]<0){
      x[i] = alpha*x[i];}
  }
}

void keras2c_PReLU(float x[], size_t size, float alpha[]) {
  /*  Parametric Rectified Linear Unit. */
  /*  f(x) = alpha * x for x < 0, f(x) = x for x >= 0, */
  /*  where alpha is a learned array with the same shape as x. */
  /*  x is overwritten by the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i]<0.0f) {
      x[i] = x[i]*alpha[i];}
  }
}
  

void keras2c_ELU(float x[], size_t size, float alpha) {
  /* Exponential Linear Unit activation (ELU) */
  /*   y = {x if x>0 */
  /* 	 alpha*(e^x - 1) else} */
  /* x is overwritten with the activated values */
    
  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = alpha*(exp(x[i])-1.0f);
    }
  }
}

void keras2c_ThresholdedReLU(float x[], size_t size, float theta) {
  /* Thresholded Rectified Linear Unit. */
  /*   f(x) = x for x > theta, f(x) = 0 otherwise. */
  /* x is overwritten with the activated values */

  for (size_t i=0; i<size; i++) {
    if (x[i]<= theta) {
      x[i] = 0;}
  }
}


void keras2c_ReLU(float x[], size_t size, float max_value, float negative_slope,
	  float threshold) {
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

#endif /* KERAS2C_ACTIVATIONS_H */
