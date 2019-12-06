#pragma once
#include <stddef.h>

#define K2C_MAX_NDIM 5

struct k2c_tensor
{
  float *array;
  size_t ndim;
  size_t numel;
  size_t shape[K2C_MAX_NDIM];
};
typedef struct k2c_tensor k2c_tensor;
