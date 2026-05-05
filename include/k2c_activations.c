/**
k2c_activations.c
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c
 */


#include <math.h>
#include <stdio.h>
#include "k2c_include.h"


/**
 * Linear activation function.
 *   y=x
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_linear_func(float * x, const size_t size) {

}
k2c_activationType * k2c_linear = k2c_linear_func;


/**
 * Exponential activation function.
 *   y = exp(x)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_exponential_func(float * x, const size_t size) {

    for (size_t i=0; i<size; ++i) {
        x[i] = expf(x[i]);
    }
}
k2c_activationType * k2c_exponential = k2c_exponential_func;


/**
 * ReLU activation function.
 *   y = max(x,0)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_relu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
k2c_activationType * k2c_relu = k2c_relu_func;


/**
 * Hard sigmoid activation function.
 *   y = clip(x+3, 0, 6) / 6
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_hard_sigmoid_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        float val = x[i] + 3.0f;
        if (val <= 0.0f) {
            x[i] = 0.0f;
        }
        else if (val >= 6.0f) {
            x[i] = 1.0f;
        }
        else {
            x[i] = val / 6.0f;
        }
    }
}
k2c_activationType * k2c_hard_sigmoid = k2c_hard_sigmoid_func;


/**
 * Tanh activation function.
 *   y = tanh(x)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_tanh_func(float * x, const size_t size) {

    for (size_t i=0; i<size; ++i) {
        x[i] = tanhf(x[i]);
    }
}
k2c_activationType * k2c_tanh = k2c_tanh_func;


/**
 * Sigmoid activation function.
 *   y = 1/(1+exp(-x))
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_sigmoid_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        x[i] = 1/(1+expf(-x[i]));
    }
}
k2c_activationType * k2c_sigmoid = k2c_sigmoid_func;

/**
 * swish activation function.
 *   y = x * (1/(1+exp(-x)))
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_swish_func(float * x, const size_t size) {

    for (size_t i = 0; i < size; ++i) {
        float xv = x[i];
        float v = xv;
        if (v < -30.0f) v = -30.0f; // Clamp to avoid overflow
        x[i] = xv / (1.0f + expf(-v));
    }
}
k2c_activationType * k2c_swish = k2c_swish_func;


/**
 * Soft max activation function.
 *   z[i] = exp(x[i]-max(x))
 *   y = z/sum(z)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_softmax_func(float * x, const size_t size) {

    float xmax = x[0];
    float sum = 0;
    for (size_t i=0; i < size; ++i) {
        if (x[i]>xmax) {
            xmax = x[i];
        }
    }

    for (size_t i=0; i < size; ++i) {
        x[i] = expf(x[i]-xmax);
    }

    for (size_t i=0; i < size; ++i) {
        sum += x[i];
    }

    sum = 1.0f/sum;
    for (size_t i=0; i < size; ++i) {
        x[i] = x[i]*sum;
    }
}
k2c_activationType * k2c_softmax = k2c_softmax_func;


/**
 * Soft plus activation function.
 *   y = ln(1+exp(x))
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_softplus_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        x[i] = log1pf(expf(x[i]));
    }
}
k2c_activationType * k2c_softplus = k2c_softplus_func;


/**
 * Soft sign activation function.
 *   y = x/(1+|x|)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_softsign_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        x[i] = x[i]/(1.0f + fabsf(x[i]));
    }
}
k2c_activationType * k2c_softsign = k2c_softsign_func;


/**
 * Leaky version of a Rectified Linear Unit.
 * It allows a small gradient when the unit is not active:
 *   y = {negative_slope*x    if x < 0}
 *       {x          if x >= 0}
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 * :param negative_slope: slope of negative portion of activation curve.
 */
void k2c_LeakyReLU(float * x, const size_t size, const float negative_slope) {

    for (size_t i=0; i<size; ++i) {
        if (x[i]<0) {
            x[i] = negative_slope*x[i];
        }
    }
}


/**
 * Parametric Rectified Linear Unit.
 * It allows a small gradient when the unit is not active:
 *   y = {alpha*x    if x < 0}
 *       {x          if x >= 0}
 * Where alpha is a learned array with the same shape as x.
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 * :param alpha: slope of negative portion of activation curve for each unit.
 */
void k2c_PReLU(float * x, const size_t size, const float * alpha) {

    for (size_t i=0; i<size; ++i) {
        if (x[i]<0.0f) {
            x[i] = x[i]*alpha[i];
        }
    }
}


/**
 * Exponential Linear Unit activation (ELU).
 *   y = {alpha*(exp(x) - 1)  if x <  0}
 *       {x                   if x >= 0}
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 * :param alpha: slope of negative portion of activation curve.
 */
void k2c_ELU(float * x, const size_t size, const float alpha) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] <= 0.0f) {
            x[i] = alpha*expm1f(x[i]);
        }
    }
}


/**
 * Thresholded Rectified Linear Unit.
 *   y = {x    if x >  theta}
         {0    if x <= theta}
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 * :param theta: threshold for activation.
 */
void k2c_ThresholdedReLU(float * x, const size_t size, const float theta) {

    for (size_t i=0; i<size; ++i) {
        if (x[i]<= theta) {
            x[i] = 0;
        }
    }
}

/**
 * Rectified Linear Unit activation function.
 *   y = {max_value       if          x >= max_value}
 *       {x               if theta <= x <  max_value}
 *       {alpha*(x-theta) if          x < theta}
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 * :param max_value: maximum value for activated x.
 * :param alpha: slope of negative portion of activation curve.
 * :param theta: threshold for activation.
 */
void k2c_ReLU(float * x, const size_t size, const float max_value,
              const float alpha, const float theta) {

    for (size_t i=0; i<size; ++i) {
        float val = x[i];
        val = val < theta ? alpha*(val - theta) : val;
        val = val > max_value ? max_value : val;
        x[i] = val;
    }
}


/**
 * SELU (Scaled Exponential Linear Unit) activation function.
 *   y = scale * (x          if x >= 0)
 *   y = scale * (alpha*(exp(x)-1) if x <  0)
 *   scale = 1.0507009873554805, alpha = 1.6732632423543772
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_selu_func(float * x, const size_t size) {

    const float alpha = 1.6732632423543772f;
    const float scale = 1.0507009873554805f;
    for (size_t i=0; i < size; ++i) {
        if (x[i] >= 0.0f) {
            x[i] = scale * x[i];
        }
        else {
            x[i] = scale * alpha * expm1f(x[i]);
        }
    }
}
k2c_activationType * k2c_selu = k2c_selu_func;


/**
 * ELU (Exponential Linear Unit) activation function (default alpha=1.0).
 *   y = x              if x >= 0
 *   y = exp(x) - 1     if x <  0
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_elu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] < 0.0f) {
            x[i] = expm1f(x[i]);
        }
    }
}
k2c_activationType * k2c_elu = k2c_elu_func;


/**
 * GELU (Gaussian Error Linear Unit) activation function.
 *   y = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_gelu_func(float * x, const size_t size) {

    const float sqrt2_inv = 0.7071067811865475f;
    for (size_t i=0; i < size; ++i) {
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * sqrt2_inv));
    }
}
k2c_activationType * k2c_gelu = k2c_gelu_func;


/**
 * Hard SiLU (Hard Swish) activation function.
 *   y = x * hard_sigmoid(x) = x * clip(x+3, 0, 6) / 6
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_hard_silu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        float val = x[i] + 3.0f;
        if (val <= 0.0f) {
            x[i] = 0.0f;
        }
        else if (val >= 6.0f) {
            /* hard_sigmoid = 1, so x * 1 = x; leave unchanged */
        }
        else {
            x[i] = x[i] * val / 6.0f;
        }
    }
}
k2c_activationType * k2c_hard_silu = k2c_hard_silu_func;


/**
 * Mish activation function.
 *   y = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_mish_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        float sp = x[i] > 20.0f ? x[i] : log1pf(expf(x[i]));
        x[i] = x[i] * tanhf(sp);
    }
}
k2c_activationType * k2c_mish = k2c_mish_func;


/**
 * ReLU6 activation function.
 *   y = min(max(x, 0), 6)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_relu6_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
        else if (x[i] > 6.0f) {
            x[i] = 6.0f;
        }
    }
}
k2c_activationType * k2c_relu6 = k2c_relu6_func;


/**
 * Log-softmax activation function.
 *   y[i] = x[i] - log(sum(exp(x)))
 * Computed in a numerically stable manner.
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_log_softmax_func(float * x, const size_t size) {

    float xmax = x[0];
    for (size_t i=1; i < size; ++i) {
        if (x[i] > xmax) {
            xmax = x[i];
        }
    }
    float sum = 0.0f;
    for (size_t i=0; i < size; ++i) {
        sum += expf(x[i] - xmax);
    }
    float log_sum = logf(sum);
    for (size_t i=0; i < size; ++i) {
        x[i] = (x[i] - xmax) - log_sum;
    }
}
k2c_activationType * k2c_log_softmax = k2c_log_softmax_func;


/**
 * Leaky ReLU activation function (default negative_slope=0.2).
 *   y = x                if x >= 0
 *   y = 0.2 * x          if x <  0
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_leaky_relu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] < 0.0f) {
            x[i] = 0.2f * x[i];
        }
    }
}
k2c_activationType * k2c_leaky_relu = k2c_leaky_relu_func;


/**
 * CELU (Continuously Differentiable ELU) activation function (default alpha=1.0).
 *   y = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
 *   With alpha=1.0: same as ELU with alpha=1.0.
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_celu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] < 0.0f) {
            x[i] = expm1f(x[i]);
        }
    }
}
k2c_activationType * k2c_celu = k2c_celu_func;


/**
 * Hard tanh activation function.
 *   y = clip(x, -1, 1)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_hard_tanh_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] < -1.0f) {
            x[i] = -1.0f;
        }
        else if (x[i] > 1.0f) {
            x[i] = 1.0f;
        }
    }
}
k2c_activationType * k2c_hard_tanh = k2c_hard_tanh_func;


/**
 * Hard shrink activation function (default lambda=0.5).
 *   y = x    if |x| > 0.5
 *   y = 0    otherwise
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_hard_shrink_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] > -0.5f && x[i] < 0.5f) {
            x[i] = 0.0f;
        }
    }
}
k2c_activationType * k2c_hard_shrink = k2c_hard_shrink_func;


/**
 * Soft shrink activation function (default lambda=0.5).
 *   y = x - 0.5   if x >  0.5
 *   y = x + 0.5   if x < -0.5
 *   y = 0         otherwise
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_soft_shrink_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        if (x[i] > 0.5f) {
            x[i] = x[i] - 0.5f;
        }
        else if (x[i] < -0.5f) {
            x[i] = x[i] + 0.5f;
        }
        else {
            x[i] = 0.0f;
        }
    }
}
k2c_activationType * k2c_soft_shrink = k2c_soft_shrink_func;


/**
 * Squareplus activation function (default b=4).
 *   y = (x + sqrt(x^2 + b)) / 2
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_squareplus_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
        x[i] = 0.5f * (x[i] + sqrtf(x[i] * x[i] + 4.0f));
    }
}
k2c_activationType * k2c_squareplus = k2c_squareplus_func;


/**
 * Sparse plus activation function.
 *   y = 0                              if x <= -sqrt(e)
 *   y = (x + sqrt(e))^2 / (4*sqrt(e)) if -sqrt(e) < x < sqrt(e)
 *   y = x                              if x >= sqrt(e)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void k2c_sparse_plus_func(float * x, const size_t size) {

    const float sqrte = 1.6487212707f;
    const float inv_4sqrte = 1.0f / (4.0f * sqrte);
    for (size_t i=0; i < size; ++i) {
        if (x[i] <= -sqrte) {
            x[i] = 0.0f;
        }
        else if (x[i] < sqrte) {
            float t = x[i] + sqrte;
            x[i] = t * t * inv_4sqrte;
        }
    }
}
k2c_activationType * k2c_sparse_plus = k2c_sparse_plus_func;
