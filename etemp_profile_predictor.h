#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
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



extern void etemp_profile_predictor(k2c_tensor* input_thomson_temp_EFITRT1_input, k2c_tensor* input_thomson_dens_EFITRT1_input, k2c_tensor* input_past_pinj_input, k2c_tensor* input_past_curr_input, k2c_tensor* input_past_tinj_input, k2c_tensor* input_past_gasA_input, k2c_tensor* input_future_pinj_input, k2c_tensor* input_future_curr_input, k2c_tensor* input_future_tinj_input, k2c_tensor* input_future_gasA_input, k2c_tensor* target_temp_output, k2c_tensor* target_dens_output);
extern void etemp_profile_predictor_initialize(); 
extern void etemp_profile_predictor_terminate();
