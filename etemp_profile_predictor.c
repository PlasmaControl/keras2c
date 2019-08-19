#ifndef ETEMP_PROFILE_PREDICTOR_H 
#define ETEMP_PROFILE_PREDICTOR_H 
#include <stdio.h> 
#include <stddef.h> 
#include <math.h> 
#include <string.h> 
#include <stdarg.h> 
#include "etemp_profile_predictor.h" 


static void k2c_linear(float x[], size_t size){
  /* linear activation. Doesn't do anything, just a dummy fn */

}

static void k2c_relu(float x[], size_t size) {
  /* Rectified Linear Unit activation (ReLU) */
  /*   y = max(x,0) */
  /* x is overwritten with the activated values */

  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = 0.0f;
    }
  }
}

static void k2c_hard_sigmoid(float x[], size_t size) {
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

static void k2c_tanh(float x[], size_t size) {
  /* standard tanh activation */
  /* x is overwritten with the activated values */
  for (size_t i=0; i<size; i++){
    x[i] = tanh(x[i]);
  }
}

static void k2c_sigmoid(float x[], size_t size) {
  /* Sigmoid activation */
  /*   y = 1/(1+exp(-x)) */
  /* x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = 1/(1+exp(-x[i]));
  }
}

static void k2c_matmul(float C[], float A[], float B[], size_t outrows,
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
    for (size_t j = 0;  j < outcols; j++) {
      for (size_t k = 0; k < innerdim; k++) {
  	C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
      }
    }
  }
}

static void k2c_affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
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

static size_t k2c_sub2idx(size_t sub[], size_t shape[], size_t ndim) {
  /* converts from subscript to linear indices in row major order */

  size_t idx = 0;
  size_t temp = 0;
  for (size_t i=0; i<ndim; i++) {
    temp = sub[i];
    for (size_t j=ndim-1; j>i; j--) {
      temp *= shape[j];}
    idx += temp;
  }
  return idx;
}

static void k2c_idx2sub(size_t idx, size_t sub[], size_t shape[], size_t ndim) {

  size_t idx2 = idx;
  for (int i=ndim-1; i>=0; i--) {
    sub[i] = idx2%shape[i];
    idx2 /= shape[i];
  }
}

static void k2c_bias_add(k2c_tensor* A, k2c_tensor* b) {
  /* adds bias vector b to tensor A. Assumes b is a rank 1 tensor */
  /* that is added to the last dimension of A */
  for (size_t i=0; i<A->numel; i+=b->numel) {
    for (size_t j=0; j<b->numel; j++) {
      A->array[i+j] += b->array[j];}
  }
}

static void k2c_dot(k2c_tensor* C, k2c_tensor* A, k2c_tensor* B, size_t axesA[],
	     size_t axesB[], size_t naxes, int normalize, float fwork[]) {

  size_t permA[K2C_MAX_NDIM];
  size_t permB[K2C_MAX_NDIM];
  size_t prod_axesA = 1;
  size_t prod_axesB = 1;
  size_t free_axesA, free_axesB;
  size_t freeA[K2C_MAX_NDIM];
  size_t freeB[K2C_MAX_NDIM];
  size_t count;
  int isin;
  size_t i,j;
  size_t newshpA[K2C_MAX_NDIM];
  size_t newshpB[K2C_MAX_NDIM];
  size_t ndimA = A->ndim;
  size_t ndimB = B->ndim;
  float *reshapeA = &fwork[0];   // temp working storage
  float *reshapeB = &fwork[A->numel];

  // find which axes are free (ie, not being summed over)
  count=0;
  for (i=0; i<A->ndim; i++) {
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
  count=0;
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

    // number of elements in inner dimension
  for (i=0; i < naxes; i++) {
    prod_axesA *= A->shape[axesA[i]];}
  for (i=0; i < naxes; i++) {
    prod_axesB *= B->shape[axesB[i]];}
  // number of elements in free dimension
  free_axesA = A->numel/prod_axesA;
  free_axesB = B->numel/prod_axesB;
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
  for (i=0; i<A->ndim; i++) {
    newshpA[i] = A->shape[permA[i]];
  }
  for (i=0; i<B->ndim; i++) {
    newshpB[i] = B->shape[permB[i]];
  }

  // reshape arrays
  for (i=0; i<A->numel; i++) {
    k2c_idx2sub(i,Asub,A->shape,ndimA);
    for (j=0; j<ndimA; j++) {
      Bsub[j] = Asub[permA[j]];}
    bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
    reshapeA[bidx] = A->array[i];
  }

  for (i=0; i<B->numel; i++) {
    k2c_idx2sub(i,Bsub,B->shape,ndimB);
    for (j=0; j<ndimB; j++) {
      Asub[j] = Bsub[permB[j]];}
    bidx = k2c_sub2idx(Asub,newshpB,ndimB);
    reshapeB[bidx] = B->array[i];
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
 
  k2c_matmul(C->array, reshapeA, reshapeB, free_axesA,
	     free_axesB, prod_axesA);
}



static void k2c_dense(k2c_tensor* output, k2c_tensor* input, k2c_tensor* kernel,
	       k2c_tensor* bias, void (*activation) (float[], size_t),
	       float fwork[]){

  if (input->ndim <=2) {
    size_t outrows;

    if (input->ndim>1) {
      outrows = input->shape[0];}
    else {
      outrows = 1;}
    size_t outcols = kernel->shape[1];
    size_t innerdim = kernel->shape[0];
    size_t outsize = outrows*outcols;
    k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
		      outrows,outcols,innerdim);
    activation(output->array,outsize);
  }
  else {
    size_t axesA[1] = {input->ndim-1};
    size_t axesB[1] = {0};
    size_t naxes = 1;
    int normalize = 0;

    k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
    k2c_bias_add(output, bias);
    activation(output->array, output->numel);
  }
}

static void k2c_reshape(k2c_tensor* input, size_t newshp[], size_t newndim) {
  for (size_t i=0; i<input->ndim; i++) {
    input->shape[i] = 1;}
  for (size_t i=0; i<newndim; i++) {
    input->shape[i] = newshp[i];}
  input->ndim = newndim;
}

static void k2c_concatenate(k2c_tensor* output, size_t axis, size_t num_tensors,...) {
/*  Concatenation several tensors. */
/* num_tensors is the total number of tensors */
/* axis is the axis along which to concatenate */
/* results are stored in output array */
/* takes variable number of tensors as inputs: */
/* concatenate(output, num_tensors, axis, tensor1, tensor2,...tensorN) etc */

  va_list args;
  k2c_tensor* arrptr;
  size_t  offset = 0;
  size_t outidx;
  size_t insub[K2C_MAX_NDIM], outsub[K2C_MAX_NDIM];
  va_start (args, num_tensors);     

  for (size_t i=0; i<num_tensors; i++) {
    arrptr = va_arg(args, k2c_tensor*);
    for (size_t j=0; j<arrptr->numel; j++) {
      k2c_idx2sub(j,insub,arrptr->shape,arrptr->ndim);
      memcpy(outsub,insub,K2C_MAX_NDIM*sizeof(size_t));
      outsub[axis] += offset;
      outidx = k2c_sub2idx(outsub,output->shape, output->ndim);
      output->array[outidx] = arrptr->array[j];
    }
    offset += arrptr->shape[axis];
  }
  va_end (args);               
}

static void k2c_pad1d(k2c_tensor* output, k2c_tensor* input, float fill,
	       size_t pad[]) {
  /* pads array in height dimension. 
     fill = fill value
     pad_top: number of rows of fill to concatenate on top of input
     pad_bottom: number of rows of fill to cat on bottom of input */

  size_t in_width = input->shape[1];
  size_t pad_top = pad[0];

  // set output array to fill value
  if (fabs(fill) < 1e-6) {
    // fill is ~zero, use memset
    memset(output->array,0,output->numel*sizeof(output->array[0]));
  }
  else {
    for(size_t i=0; i<output->numel; i++) {
      output->array[i] = fill;}
  }

  // memcpy the old array in the right place
  size_t offset = pad_top*in_width;
  memcpy(&output->array[offset],&input->array[0],
	 input->numel*sizeof(input->array[0]));
}

static void k2c_conv1d(k2c_tensor* output, k2c_tensor* input, k2c_tensor* kernel,
		k2c_tensor* bias, size_t stride, size_t dilation,
		   void (*activation) (float[], size_t)) {
  /* 1D (temporal) convolution. Assumes a "channels last" structure
   */
  memset(output->array,0,output->numel*sizeof(output->array[0]));

  size_t out_times = output->shape[0];
  size_t out_channels = output->shape[1];
  size_t in_channels = input->shape[1];

  for (size_t x0=0; x0 < out_times; x0++){
    for (size_t k=0; k < out_channels; k++) {
      for (size_t z=0; z < kernel->shape[0]; z++) {
	for (size_t q=0; q < in_channels; q++) {
	  /* size_t outsub[K2C_MAX_NDIM] = {x0,k}; */
	  /* size_t inpsub[K2C_MAX_NDIM] = {x0*stride + dilation*z,q}; */
	  /* size_t kersub[K2C_MAX_NDIM] = {z,q,k}; */


	  /* size_t outidx = x0*out_channels + k; */
	  /* size_t inpidx = (x0*stride + dilation*z)*in_channels + q; */
	  /* size_t keridx = z*(kernel->shape[2]*kernel->shape[1]) + q*(kernel->shape[1]) + k; */
	  
	  output->array[x0*out_channels + k] +=
	    kernel->array[z*(kernel->shape[2]*kernel->shape[1]) + q*(kernel->shape[1]) + k]*
	    input->array[(x0*stride + dilation*z)*in_channels + q];
	  /* output->array[k2c_sub2idx(outsub,output->shape,output->ndim)] += */
	  /*   kernel->array[k2c_sub2idx(kersub,kernel->shape,kernel->ndim)]* */
	  /*   input->array[k2c_sub2idx(inpsub,input->shape,input->ndim)]; */
	}
      }
    }
  }
  k2c_bias_add(output,bias);
  activation(output->array,output->numel);
}

static void k2c_lstmcell(float state[], float input[], k2c_tensor* kernel,
		  k2c_tensor* recurrent_kernel, k2c_tensor* bias, float fwork[],
		  void (*recurrent_activation) (float*, size_t),
		  void (*output_activation)(float*, size_t)){


  size_t units = recurrent_kernel->shape[1];
  size_t in_width = kernel->shape[0]/4;
  
  float *h_tm1 = &state[0];  // previous memory state
  float *c_tm1 = &state[units];  // previous carry state
  size_t outrows = 1;
  float *Wi = &kernel->array[0];
  float *Wf = &kernel->array[in_width*units];
  float *Wc = &kernel->array[2*in_width*units];
  float *Wo = &kernel->array[3*in_width*units];
  float *Ui = &recurrent_kernel->array[0];
  float *Uf = &recurrent_kernel->array[units*units];
  float *Uc = &recurrent_kernel->array[2*units*units];
  float *Uo = &recurrent_kernel->array[3*units*units];
  float *bi = &bias->array[0];
  float *bf = &bias->array[units];
  float *bc = &bias->array[2*units];
  float *bo = &bias->array[3*units];
  float *xi = &fwork[0];
  float *xf = &fwork[units];
  float *xc = &fwork[2*units];
  float *xo = &fwork[3*units];
  float *yi = &fwork[4*units];
  float *yf = &fwork[5*units];
  float *yc = &fwork[6*units];
  float *yo = &fwork[7*units];
 

  //xi = input*Wi + bi;
  k2c_affine_matmul(xi, input, Wi, bi, outrows, units, in_width);
  //xf = input*Wf + bf;
  k2c_affine_matmul(xf, input, Wf, bf, outrows, units, in_width);
  //xc = input*Wc + bc;
  k2c_affine_matmul(xc, input, Wc, bc, outrows, units, in_width);
  //xo = input*Wo + bo;
  k2c_affine_matmul(xo, input, Wo, bo, outrows, units, in_width);
 
  // yi = recurrent_activation(xi + h_tm1*Ui);
  k2c_affine_matmul(yi, h_tm1, Ui, xi, outrows, units, units);
  recurrent_activation(yi, units);

  // yf = recurrent_activation(xf + h_tm1*Uf); 
  k2c_affine_matmul(yf, h_tm1, Uf, xf, outrows, units, units);
  recurrent_activation(yf, units);

  // yc = yf.*c_tm1 + yi.*output_activation(xc + h_tm1*Uc);
  k2c_affine_matmul(yc, h_tm1, Uc, xc, outrows, units, units);
  output_activation(yc, units);
  for (size_t i=0; i < units; i++){
    yc[i] = yf[i]*c_tm1[i] + yi[i]*yc[i];}

  
  // yo = recurrent_activation(xo + h_tm1*Uo); 
  k2c_affine_matmul(yo, h_tm1, Uo, xo, outrows, units, units);
  recurrent_activation(yo, units);

  // h = yo.*output_activation(yc); 
  // state = [h;yc];
  for (size_t i=0; i < units; i++){
    state[units+i] = yc[i];}

  output_activation(yc, units);

  for (size_t i=0; i < units; i++){
    state[i] = yo[i]*yc[i];}

}

static void k2c_lstm(k2c_tensor* output, k2c_tensor* input, float state[],
	      k2c_tensor* kernel, k2c_tensor* recurrent_kernel,
	      k2c_tensor* bias, float fwork[], int go_backwards,
	      int return_sequences,
	      void (*recurrent_activation) (float*, size_t),
	      void (*output_activation)(float*, size_t)){


  size_t in_height = input->shape[0];
  size_t in_width = input->shape[1];
  size_t units = recurrent_kernel->shape[1];
  if (go_backwards) {
    for (int i=in_height-1; i>-1; i--) {
      k2c_lstmcell(state, &input->array[i*in_width], kernel, recurrent_kernel,
		   bias, fwork, recurrent_activation, output_activation);
      if (return_sequences) {
	for (size_t j=0; j<units; j++) {
	  output->array[(in_height-1-i)*units+j] = state[j];}
      }
    }
  }
  else{   
    for (size_t i=0; i < in_height; i++){
      k2c_lstmcell(state, &input->array[i*in_width], kernel, recurrent_kernel,
		   bias, fwork, recurrent_activation, output_activation);
      if (return_sequences) {
	for (size_t j=0; j<units; j++) {
	  output->array[i*units+j] = state[j];}
      }
    }
  }
  if (!return_sequences) {
    for (size_t i=0; i < units; i++){
      output->array[i] = state[i];}
  }
}

 
void etemp_profile_predictor(k2c_tensor* input_thomson_temp_EFITRT1_input, k2c_tensor* input_thomson_dens_EFITRT1_input, k2c_tensor* input_past_pinj_input, k2c_tensor* input_past_curr_input, k2c_tensor* input_past_tinj_input, k2c_tensor* input_past_gasA_input, k2c_tensor* input_future_pinj_input, k2c_tensor* input_future_curr_input, k2c_tensor* input_future_tinj_input, k2c_tensor* input_future_gasA_input, k2c_tensor* target_temp_output, k2c_tensor* target_dens_output) { 

size_t reshape_5_newndim = 2; 
size_t reshape_5_newshp[K2C_MAX_NDIM] = {8,1,1,1,1}; 


size_t reshape_7_newndim = 2; 
size_t reshape_7_newshp[K2C_MAX_NDIM] = {8,1,1,1,1}; 


size_t reshape_9_newndim = 2; 
size_t reshape_9_newshp[K2C_MAX_NDIM] = {8,1,1,1,1}; 


size_t reshape_11_newndim = 2; 
size_t reshape_11_newshp[K2C_MAX_NDIM] = {8,1,1,1,1}; 


size_t reshape_4_newndim = 2; 
size_t reshape_4_newshp[K2C_MAX_NDIM] = {4,1,1,1,1}; 


size_t reshape_6_newndim = 2; 
size_t reshape_6_newshp[K2C_MAX_NDIM] = {4,1,1,1,1}; 


size_t reshape_8_newndim = 2; 
size_t reshape_8_newshp[K2C_MAX_NDIM] = {4,1,1,1,1}; 


size_t reshape_10_newndim = 2; 
size_t reshape_10_newshp[K2C_MAX_NDIM] = {4,1,1,1,1}; 


size_t reshape_1_newndim = 3; 
size_t reshape_1_newshp[K2C_MAX_NDIM] = { 1,33, 1, 1, 1}; 


size_t reshape_2_newndim = 3; 
size_t reshape_2_newshp[K2C_MAX_NDIM] = { 1,33, 1, 1, 1}; 


size_t concatenate_3_num_tensors0 = 4; 
size_t concatenate_3_axis = -2; 
float concatenate_3_output_array[32] = {0}; 
k2c_tensor concatenate_3_output = {&concatenate_3_output_array[0],2,32,{8,4,1,1,1}}; 


size_t concatenate_2_num_tensors0 = 4; 
size_t concatenate_2_axis = -2; 
float concatenate_2_output_array[16] = {0}; 
k2c_tensor concatenate_2_output = {&concatenate_2_output_array[0],2,16,{4,4,1,1,1}}; 


size_t concatenate_1_num_tensors0 = 2; 
size_t concatenate_1_axis = -2; 
float concatenate_1_output_array[66] = {0}; 
k2c_tensor concatenate_1_output = {&concatenate_1_output_array[0],3,66,{ 1,33, 2, 1, 1}}; 


float lstm_1_output_array[33] = {0}; 
k2c_tensor lstm_1_output = {&lstm_1_output_array[0],1,33,{33, 1, 1, 1, 1}}; 
float lstm_1_fwork[264] = {0}; 
int lstm_1_go_backwards = 0;
int lstm_1_return_sequences = 0;
float lstm_1_state[66] = {0}; 
float lstm_1_kernel_array[528] = {
1.66432470e-01,-6.37671649e-01,-3.86500321e-02,1.13995388e-01,-2.54238695e-01,
-2.31675595e-01,-9.38857943e-02,-5.87346375e-01,5.61501980e-01,-1.79934427e-01,
4.87425983e-01,-1.43358484e-01,-2.28659362e-01,-3.26810062e-01,-1.02358490e-01,
-2.04169497e-01,-3.27583134e-01,3.13748062e-01,-6.60343394e-02,-2.64971554e-02,
-4.07174766e-01,-4.08469476e-02,-2.35934153e-01,-1.75013334e-01,-4.18117046e-01,
-3.79534811e-01,2.85232991e-01,-7.13289483e-03,-6.02314472e-01,-4.73195493e-01,
1.66718706e-01,2.23380458e-02,-1.19184554e-01,2.79073566e-01,-2.65981019e-01,
5.55483878e-01,-9.00770649e-02,-3.12893331e-01,1.42117262e-01,9.26874354e-02,
-9.36640739e-01,1.00611903e-01,1.83176890e-01,6.18591964e-01,-2.41301626e-01,
8.61674249e-02,4.27419543e-01,7.77865648e-02,1.32850766e-01,-3.30643922e-01,
-3.61274958e-01,1.20870344e-01,-1.73445046e-01,1.43095717e-01,-1.85626373e-01,
-4.09229174e-02,3.02105844e-01,-2.01613903e-01,5.64533353e-01,1.15306103e+00,
1.52688078e-03,2.48132333e-01,3.25985104e-01,1.40821606e-01,-1.28617868e-01,
3.69697422e-01,2.49548480e-01,1.09289326e-01,-8.36971626e-02,-1.02627855e-02,
-8.59805793e-02,-1.81702122e-01,7.13753328e-02,-4.39686298e-01,-6.32937253e-02,
-2.52011150e-01,1.59953222e-01,-2.55385134e-02,-6.11171722e-02,-6.60151005e-01,
-2.76978701e-01,-1.05445227e-02,-3.40016454e-01,-4.42002565e-01,-8.35998431e-02,
5.48587814e-02,-6.55554771e-01,-2.08296061e-01,-1.84817120e-01,-2.72849917e-01,
-6.67054176e-01,-1.03024971e+00,1.87219188e-01,-8.36711898e-02,-4.66055237e-02,
-5.33309817e-01,-1.34639367e-01,-5.08977147e-03,-3.74547899e-01,1.06175259e-01,
-1.07181519e-01,-1.79430589e-01,-5.64699709e-01,5.69587871e-02,4.92108762e-02,
1.41665965e-01,-4.58820015e-01,1.78467497e-01,3.09835244e-02,-9.86286998e-02,
1.59849361e-01,3.61078009e-02,1.66298091e-01,1.23421371e-01,-2.04615846e-01,
1.00551642e-01,2.08449200e-01,1.31817594e-01,5.36660925e-02,4.00136173e-01,
-2.82944649e-01,1.25965387e-01,-4.51524526e-01,2.54574835e-01,1.43102407e-01,
8.35636914e-01,1.17412955e-01,-1.90444477e-02,-5.64662516e-02,6.10205010e-02,
3.57418001e-01,1.09490000e-01,1.56165570e-01,-4.79341507e-01,-2.17310414e-01,
-1.22005828e-01,-2.81425476e-01,-6.88236713e-01,-3.31091434e-02,-1.22297101e-01,
6.55491948e-01,-5.32864749e-01,-4.12979722e-01,-3.76167297e-02,-4.15259182e-01,
-6.94360435e-01,-2.74822265e-02,-1.69494376e-01,3.58369708e-01,1.48516819e-01,
6.99785948e-02,-1.33688182e-01,9.39918086e-02,-8.11117664e-02,-2.23924756e-01,
3.34227920e-01,-3.29027504e-01,4.10448462e-01,-4.95161921e-01,-2.05011621e-01,
-2.94231325e-01,6.44669607e-02,6.61877841e-02,3.79587948e-01,-2.51350850e-01,
3.60885888e-01,-2.03773335e-01,5.09735942e-01,-1.47496358e-01,-3.86402041e-01,
-2.27474198e-01,1.41117439e-01,5.55845089e-02,4.80241105e-02,-3.75075191e-01,
2.70775974e-01,3.85642827e-01,1.14095367e-01,-5.95900929e-04,5.26877306e-02,
-1.38001412e-01,-2.33595148e-01,6.22473836e-01,2.35787928e-02,-2.64515281e-02,
5.57074606e-01,-1.66574672e-01,-1.87149540e-01,-1.15063675e-01,-1.03504926e-01,
3.63330424e-01,2.01864794e-01,-1.04545288e-01,-1.24252446e-01,1.83362082e-01,
-1.17376000e-01,-1.06668413e-01,-1.18704297e-01,-6.34906068e-02,-2.08909109e-01,
-3.31874192e-02,6.66835904e-02,-5.83874464e-01,-6.20872021e-01,1.62667289e-01,
-3.55923474e-01,4.71062884e-02,-1.05631340e+00,-3.93194646e-01,3.29912640e-02,
-5.12502007e-02,-2.71676511e-01,4.91705507e-01,-8.50516632e-02,1.60230413e-01,
-1.53851941e-01,2.07723171e-01,-1.17461495e-01,-3.54730099e-01,-5.90690002e-02,
-2.42228061e-01,1.75965980e-01,-1.23372816e-01,1.10188894e-01,-3.07817072e-01,
8.16256851e-02,-1.35268839e-02,-7.68491998e-02,-1.01290703e-01,2.40597874e-01,
-3.85939568e-01,-6.65871426e-02,-3.45033295e-02,-5.79352211e-03,-3.29874963e-01,
6.55850232e-01,1.72094777e-01,-1.31054193e-01,1.86006591e-01,-2.90671676e-01,
4.98598009e-01,2.45349526e-01,-2.03677773e-01,2.52742499e-01,2.33678877e-01,
1.98494405e-01,7.80894160e-02,3.01905721e-02,2.97940880e-01,-1.33705944e-01,
1.70773357e-01,-3.10877035e-03,1.81413323e-01,-1.33479014e-01,-2.50651389e-01,
-4.12655890e-01,4.63128179e-01,3.39359522e-01,-1.91067651e-01,1.12124942e-02,
-1.75839067e-01,-1.10609308e-01,1.21455938e-01,2.19899788e-01,-1.35540992e-01,
5.08854926e-01,-4.37928826e-01,-2.50489771e-01,-3.58662426e-01,-7.30386615e-01,
-5.65001788e-03,-3.64092439e-01,-1.48190171e-01,2.34524623e-01,8.87542143e-02,
-6.36679471e-01,-7.21155852e-02,-3.27926189e-01,-7.56035626e-01,-3.33055258e-02,
-1.70218363e-01,-6.33778691e-01,8.86765718e-02,8.44462141e-02,-2.60840863e-01,
-5.27714789e-01,4.57625836e-03,6.61579445e-02,-2.70774603e-01,-2.00475156e-01,
7.20344961e-01,1.60384104e-01,1.08946621e-01,-3.78740400e-01,1.81352906e-02,
-5.38365722e-01,1.15042686e-01,3.47834043e-02,-2.66049523e-03,1.99939817e-01,
-4.09803167e-02,5.41311502e-01,-3.76134552e-02,2.13395171e-02,5.59275337e-02,
-5.95186315e-02,3.80308986e-01,3.39049906e-01,-1.32486090e-01,-2.77741859e-03,
-4.23039794e-01,-9.93450265e-03,-7.47354561e-03,7.99797848e-03,1.07187614e-01,
-6.64716121e-03,1.03973635e-02,1.08855315e-01,-7.95602351e-02,1.06111504e-01,
-1.70032158e-01,6.58427030e-02,4.32068586e-01,4.84411478e-01,-1.96647458e-02,
-6.86060861e-02,-9.61053148e-02,-2.47043115e-03,7.29851611e-03,-7.49405995e-02,
3.90305936e-01,-3.95345390e-01,-9.19350535e-02,-2.23060712e-01,2.04629242e-01,
-1.89313278e-01,1.24007855e-02,-4.68662262e-01,-3.57423723e-01,2.21824497e-01,
-1.51154846e-01,-3.15141827e-01,-2.05949172e-01,-5.17177105e-01,2.70911127e-01,
1.02763452e-01,1.05739944e-01,-1.29435241e-01,3.47263515e-02,-6.98513165e-02,
-7.12405704e-03,-1.30937649e-02,5.26706735e-03,-3.62151302e-02,-9.69320685e-02,
-8.24562907e-02,1.11695111e-01,-1.55503482e-01,-2.33595669e-01,-4.80039656e-01,
6.56827586e-03,2.19195662e-03,2.98328966e-01,4.57347989e-01,2.27765545e-01,
1.31064713e-01,1.43834978e-01,-5.33590674e-01,1.31528392e-01,-2.02478021e-01,
2.83942878e-01,-1.05713392e-02,7.75796250e-02,-2.21463516e-02,-1.91659495e-01,
1.06460892e-01,4.52608392e-02,5.81949297e-03,-1.75003856e-01,6.00277297e-02,
1.96355194e-01,7.40822181e-02,-3.24192911e-01,-1.01115651e-01,4.10312772e-01,
-3.85091342e-02,4.00531918e-01,6.69777930e-01,1.53125435e-01,1.96857080e-02,
-3.14095378e-01,2.76628345e-01,-1.17058754e-01,-1.01018988e-01,3.58751155e-02,
-3.42277810e-02,4.54038352e-01,-3.08527142e-01,6.39464080e-01,2.14576885e-01,
-2.30132341e-01,5.62174737e-01,-9.43817049e-02,-1.10479139e-01,8.78785610e-01,
1.32863238e-01,2.38219127e-01,3.89943779e-01,5.44072926e-01,-5.54118752e-01,
3.62352252e-01,-6.13237545e-03,3.88776273e-01,9.71267939e-01,-7.33866692e-02,
2.35664874e-01,6.44328237e-01,-6.38414562e-01,1.25946715e-01,4.78185937e-02,
3.89723224e-03,9.52445805e-01,-2.30692789e-01,1.82459429e-02,1.73032254e-01,
-5.05272865e-01,-1.20089017e-01,2.92325377e-01,9.26084071e-02,3.58332455e-01,
-1.89894557e-01,-5.23712765e-03,-2.87499875e-01,-2.28831857e-01,-2.48111993e-01,
1.74494803e-01,-7.85158217e-01,2.98200905e-01,2.19044670e-01,4.12462473e-01,
-3.50749344e-01,-1.35352582e-01,1.15935273e-01,-9.02131855e-01,4.77070846e-02,
-1.51566982e-01,2.98912972e-02,3.31436068e-01,9.80690718e-02,-2.23699316e-01,
-1.52758032e-01,5.14730401e-02,6.77256100e-03,3.38952169e-02,-7.81021565e-02,
9.86151993e-02,-7.25790253e-03,-2.86450535e-01,1.00769162e-01,9.76902395e-02,
-1.72027424e-02,1.05132803e-01,3.62726390e-01,-2.76961476e-01,6.81144595e-01,
3.45450252e-01,-2.20139533e-01,1.49384975e-01,-1.61890402e-01,5.81969544e-02,
1.07664526e-01,-2.33762696e-01,-2.15759471e-01,-3.83100174e-02,1.26125902e-01,
-3.79234850e-01,6.51976764e-01,4.18563262e-02,-8.86169821e-02,-1.35266691e-01,
2.15663793e-04,1.94657575e-02,-2.52770633e-01,-1.37540489e-01,-2.88568474e-02,
3.04390877e-01,-1.48292124e-01,-3.17600727e-01,-2.72310451e-02,-2.55857617e-01,
5.16238332e-01,-5.47888160e-01,-1.95487380e-01,5.86293712e-02,6.73011914e-02,
1.40101805e-01,-4.06365655e-02,5.73820733e-02,-4.25061315e-01,2.03889310e-01,
3.52505803e-01,5.76350130e-02,3.16192657e-01,-3.34707111e-01,9.55919385e-01,
4.42582816e-02,4.80746068e-02,-4.34423894e-01,3.39163870e-01,-2.38458086e-02,
-1.22998536e-01,-5.00650071e-02,3.39656532e-01,-1.27439171e-01,1.12454161e-01,
-1.95219710e-01,7.21675009e-02,9.98564512e-02,1.89571276e-01,2.85241634e-01,
6.88471943e-02,6.69366598e-01,-4.83171940e-02,-1.36672435e-02,-2.51452982e-01,
2.07605124e-01,1.16001830e-01,3.30712259e-01,}; 
k2c_tensor lstm_1_kernel = {&lstm_1_kernel_array[0],2,528,{16,33, 1, 1, 1}}; 
float lstm_1_recurrent_kernel_array[4356] = {
-1.46492481e-01,-5.02129942e-02,9.61417779e-02,1.12640515e-01,4.87510115e-01,
-7.06288218e-02,8.69655013e-02,-9.92891416e-02,-2.95035560e-02,-2.65045650e-02,
5.08895554e-02,5.62249348e-02,1.70007512e-01,5.97981215e-02,3.42310108e-02,
9.83138904e-02,-1.72030091e-01,3.07470448e-02,2.40399912e-02,-1.33570600e-02,
-1.48454651e-01,7.40861811e-04,1.11883171e-01,-3.94908279e-01,-4.78322536e-01,
-4.34965461e-01,1.05459698e-01,1.83315247e-01,-1.53198779e-01,4.63765115e-02,
1.06207080e-01,-1.49158657e-01,-2.17419304e-02,9.22516957e-02,-3.79093677e-01,
-7.70522654e-02,5.91355525e-02,2.21224595e-02,-7.74565339e-02,-1.81780070e-01,
-2.28419557e-01,2.22098500e-01,6.94532990e-02,4.34131324e-02,2.57511348e-01,
-7.17167929e-02,-2.26310156e-02,-7.89338425e-02,6.19634390e-02,-6.39133975e-02,
8.75472128e-02,-1.26330227e-01,-3.84004153e-02,-1.57678097e-01,9.42133218e-02,
-5.48678562e-02,-1.21819243e-01,-1.49224639e-01,-6.58561960e-02,-3.53570879e-02,
2.08457038e-01,-4.02390569e-01,-1.48227617e-01,7.88806081e-02,-4.07664180e-02,
-2.94998258e-01,1.01385489e-01,1.51923671e-01,6.77443296e-02,1.17599415e-02,
-4.48888503e-02,2.30520651e-01,7.95190930e-02,5.17359897e-02,9.06969421e-03,
1.22732716e-02,3.79070081e-02,-2.55959362e-01,1.11919031e-01,-2.95109808e-01,
-1.20726876e-01,-5.15405834e-02,-1.42460028e-02,8.71522631e-03,1.09010532e-01,
-1.75055534e-01,-1.10016301e-01,8.55080932e-02,-2.17125520e-01,1.61297947e-01,
-2.98860855e-02,1.54210418e-01,3.12017709e-01,-5.53971455e-02,-1.10142350e-01,
4.98746224e-02,3.54911783e-03,-2.16838583e-01,9.86893550e-02,-1.46700218e-01,
-2.31404640e-02,-2.81449914e-01,4.72467542e-02,9.38984752e-02,-1.11203656e-01,
-2.24273339e-01,9.19414759e-02,1.49945663e-02,1.52042970e-01,7.92264491e-02,
2.00169273e-02,1.53685898e-01,-3.02878767e-01,-1.34988248e-01,7.75178149e-02,
-6.53220639e-02,5.92633113e-02,6.61065131e-02,5.00189103e-02,4.00900207e-02,
-6.77687749e-02,-4.20351550e-02,2.14236751e-01,-1.39404954e-02,2.47136205e-01,
1.64818674e-01,1.00285478e-01,7.35811293e-02,-9.95609164e-02,3.92051227e-02,
-6.11161590e-02,-2.98134889e-02,5.18345237e-01,2.41866127e-01,-3.10031027e-01,
-6.00681454e-02,-4.91054326e-01,1.34757813e-02,9.08854231e-02,1.63575470e-01,
3.58329058e-01,-4.64388222e-01,-1.51687965e-01,-2.19880611e-01,-1.72281668e-01,
-3.34149748e-02,-1.11087501e-01,-7.44479373e-02,2.35972047e-01,-9.06409845e-02,
2.12684721e-02,6.74578622e-02,-3.13404471e-01,-1.27230793e-01,-3.17647830e-02,
-8.79121479e-03,6.49294332e-02,2.54686296e-01,1.94088221e-01,-7.76347965e-02,
-4.21118457e-03,5.93380965e-02,1.35121062e-01,-2.26526871e-01,-7.47227967e-02,
1.58068947e-02,1.35707811e-01,6.22479767e-02,3.80262509e-02,-7.14598820e-02,
-1.70645583e-02,-4.70143976e-03,2.09997948e-02,-2.19144180e-01,-1.18647534e-02,
-1.19929135e-01,-1.09082349e-01,1.59665525e-01,4.84575406e-02,-3.94446217e-03,
-8.35712180e-02,-3.72583032e-01,-1.95491184e-02,-2.81413719e-02,4.82794046e-02,
-2.10192055e-01,1.05576463e-01,2.04511970e-01,1.28705800e-01,-1.97735950e-01,
2.39614397e-01,9.54361930e-02,-8.21136578e-04,1.11578099e-01,2.00388022e-02,
2.12299358e-02,-1.59435540e-01,1.33243844e-01,-1.11193940e-01,-1.08243376e-01,
-4.83012162e-02,8.38973373e-02,-1.38881579e-01,-5.71414456e-02,-2.59828269e-02,
5.91613278e-02,-1.39510259e-01,-2.25601848e-02,-1.03907753e-02,6.94299638e-02,
2.81615984e-02,-2.67786570e-02,-4.17535938e-02,4.85221110e-02,9.51000005e-02,
6.44372776e-02,-3.86869460e-02,-1.43120140e-01,7.96377435e-02,-4.50745896e-02,
1.21960774e-01,-1.41942143e-01,-1.27975345e-01,-5.09099551e-02,1.13049194e-01,
8.99250507e-02,-1.29233018e-01,-9.48441774e-03,5.51566668e-02,1.29709005e-01,
3.42999995e-02,-5.81474639e-02,4.87473607e-01,-7.80882612e-02,-2.25745648e-01,
-6.54361323e-02,-3.43445837e-01,8.01462159e-02,5.48701696e-02,-2.52869636e-01,
2.30558157e-01,-3.40882945e-03,1.45352939e-02,-1.48236886e-01,-7.80403316e-02,
-6.95640966e-02,-2.30287742e-02,-2.51614213e-01,-8.32117349e-02,1.89340949e-01,
7.52864927e-02,-4.14179325e-01,1.90849900e-01,-5.10242581e-02,-3.37608874e-01,
-3.48117501e-02,2.57356435e-01,6.77812323e-02,5.39768636e-02,-3.05709481e-01,
1.10915504e-01,-1.05056101e-02,-1.61763743e-01,-3.68843004e-02,-2.12053433e-01,
2.17880961e-02,1.38601601e-01,2.84176022e-01,1.16854250e-01,-1.30570214e-02,
1.19306825e-01,2.25318179e-01,-3.49636823e-02,-2.07724690e-01,-1.29612297e-01,
1.53028041e-01,1.91827923e-01,-1.46336645e-01,1.90965738e-02,9.27447230e-02,
-1.31579623e-01,1.60250664e-01,3.61085571e-02,4.59090769e-02,-2.63222575e-01,
3.10469151e-01,-3.95491496e-02,3.14547420e-01,-2.17461344e-02,-1.95233867e-01,
-3.05028204e-02,-5.90013638e-02,-4.33432311e-02,-1.08387254e-01,-6.22329898e-02,
4.56940532e-01,-6.63568377e-02,-7.55679458e-02,3.82180870e-01,-3.87676693e-02,
-1.46732271e-01,-7.44963288e-02,3.15023251e-02,-1.20385692e-01,5.82438298e-02,
2.45975656e-03,-3.30079287e-01,6.90434575e-02,7.58770257e-02,-2.69766867e-01,
-3.32533754e-02,-6.80126948e-04,-1.67700592e-02,-1.55844152e-01,-9.03503597e-03,
-2.39087064e-02,1.65172264e-01,-1.41156673e-01,-2.32458964e-01,1.88629910e-01,
-3.05815786e-01,3.78224611e-01,-1.51704952e-01,6.60991818e-02,6.07042057e-06,
-1.59227550e-01,-1.16955109e-01,-1.85161438e-02,4.99764644e-02,-4.02394354e-01,
1.91059515e-01,1.27483279e-01,-1.15405116e-02,7.12504089e-02,1.25401765e-01,
-6.05366752e-02,7.53751099e-02,2.01149836e-01,4.28259233e-03,1.91347718e-01,
-1.16672598e-01,8.75006020e-02,-1.22683868e-01,-5.81812263e-02,-7.11277127e-02,
-1.22128427e-02,1.98493898e-02,-4.70875800e-02,1.11894660e-01,-1.49002373e-02,
-2.40814295e-02,-2.84370303e-01,5.13420478e-02,-3.90192382e-02,-5.46184648e-03,
-1.86691105e-01,1.08527049e-01,1.57957137e-01,-5.21530397e-02,1.72199935e-01,
-4.70167734e-02,-5.26389964e-02,-4.82745655e-02,-4.77464609e-02,2.22863317e-01,
1.38406828e-01,1.39539033e-01,-1.41948342e-01,-9.30686370e-02,-1.22820817e-01,
9.44981202e-02,-4.57942605e-01,4.33019027e-02,1.01942509e-01,-1.14544980e-01,
-2.52295285e-01,-8.27162042e-02,5.38872648e-03,7.45070130e-02,-1.00393869e-01,
2.38339454e-01,6.16586842e-02,-7.97907189e-02,-3.75316036e-03,-2.18247116e-01,
-7.03201890e-02,4.40631732e-02,5.86614311e-02,4.75003958e-01,-3.89411226e-02,
-8.10245574e-02,7.35073835e-02,7.33703673e-02,-7.60362819e-02,4.26111594e-02,
-2.11280897e-01,-6.33254647e-02,6.36600107e-02,-1.92674533e-01,1.37114143e-02,
2.54154265e-01,-3.30677405e-02,2.50799600e-02,-4.70103975e-03,1.68640181e-01,
5.99482991e-02,-5.41194621e-03,-2.18916163e-02,-1.42379239e-01,-1.39118552e-01,
-6.67766482e-02,1.74951240e-01,1.49542540e-01,-1.12647519e-01,-5.96260577e-02,
-8.84530786e-03,-1.69531956e-01,-1.28276357e-02,-4.52565253e-02,2.84344673e-01,
-2.86797136e-01,-1.63972639e-02,1.98351756e-01,1.14660310e-02,1.97478998e-02,
6.49348879e-03,1.00134514e-01,-4.66159910e-01,1.53293431e-01,2.60032684e-01,
9.81117412e-02,1.91102162e-01,3.85996908e-01,-3.79489399e-02,-2.18276456e-01,
-8.26326832e-02,3.46611947e-01,-1.37195393e-01,-1.29510000e-01,7.02892756e-03,
4.19372171e-02,-4.67367083e-01,2.50898689e-01,1.56133056e-01,5.28524630e-02,
-4.13281262e-01,5.06821632e-01,9.23418105e-02,-1.46178544e-01,1.87508881e-01,
1.03189856e-01,-3.94375883e-02,-1.63424611e-01,4.38986599e-01,8.38016212e-01,
1.31640837e-01,1.88117102e-01,2.40827590e-01,1.45807788e-01,3.12664174e-02,
4.76407766e-01,2.80105442e-01,-1.06936887e-01,1.58776581e-01,-1.83108002e-01,
6.14771359e-02,1.83936685e-01,-1.58543084e-02,7.66749606e-02,1.85764357e-01,
1.01017863e-01,1.82781786e-01,7.64982402e-02,-1.29128084e-01,-9.58384201e-02,
-2.68381178e-01,-6.69787377e-02,1.48874447e-01,-1.41204387e-01,-1.00776315e-01,
5.10377847e-02,1.04465768e-01,3.79919894e-02,-1.88801616e-01,1.11461468e-01,
1.16499901e-01,1.07095174e-01,-2.70135030e-02,1.93527676e-02,-7.32730329e-03,
-5.48920333e-02,-1.57079697e-02,-3.05449823e-04,-1.52408779e-01,6.15510754e-02,
-2.11088676e-02,-2.05393471e-02,-4.12997715e-02,-1.59986258e-01,5.04961722e-02,
-8.55724588e-02,1.67458802e-02,6.32823110e-02,5.60736507e-02,-1.12314731e-01,
9.39807445e-02,1.14909358e-01,-3.11665069e-02,6.01221249e-02,-1.58774450e-01,
-1.38290301e-01,1.34022757e-01,-1.08133316e-01,-5.51055446e-02,-6.17570840e-02,
-2.57952251e-02,3.80651467e-02,8.45867693e-02,4.59618792e-02,2.61546224e-02,
6.99121132e-02,1.13929659e-01,-1.95501149e-02,2.89011262e-02,5.08276820e-02,
2.43265014e-02,3.31492014e-02,5.49774505e-02,-1.58693239e-01,2.99557954e-01,
-1.12016372e-01,-7.57487416e-02,3.89215589e-01,-9.01989788e-02,3.04128807e-02,
-3.44610922e-02,2.91460246e-01,3.53812426e-01,-8.10475573e-02,7.59836063e-02,
5.89811988e-02,-3.07417184e-01,-9.88431871e-02,1.00920051e-01,-1.50627673e-01,
-3.12604398e-01,1.34328097e-01,-1.42383873e-01,-1.68139324e-01,-5.72782196e-02,
2.67619416e-02,1.95616018e-02,-2.08620921e-01,-7.35622793e-02,7.24638224e-01,
-3.29980603e-03,1.32809818e-01,1.10030904e-01,1.38282459e-02,-2.18944103e-01,
-1.28876612e-01,-1.43959243e-02,2.36495480e-01,-7.67575623e-03,1.77598760e-01,
1.44076541e-01,-4.86310907e-02,-1.68995768e-01,7.29815587e-02,1.25148714e-01,
1.80735588e-01,-5.40483445e-02,2.72066474e-01,8.59175250e-02,-2.66120017e-01,
-1.48733286e-02,1.49782389e-01,-1.03734531e-01,1.90101534e-01,-2.20481161e-04,
-1.27537876e-01,-7.68417865e-02,4.83039826e-01,1.12720383e-02,8.57578684e-03,
-2.06633657e-01,2.12915063e-01,-7.80751109e-02,2.30931178e-01,-1.18115604e-01,
6.40435889e-02,-6.81093112e-02,8.57342407e-02,4.42757495e-02,2.51323264e-02,
-4.27150466e-02,9.05454084e-02,-1.53986573e-01,2.66366126e-03,-8.14892910e-03,
9.91526991e-03,1.55343562e-01,-8.88226405e-02,-1.16298079e-01,-3.52846161e-02,
-4.23164368e-02,1.34047745e-02,-5.46911098e-02,-1.26594026e-02,3.11277956e-02,
1.23293512e-02,1.33019378e-02,-6.67832047e-02,3.80141549e-02,2.35251151e-02,
1.10035077e-01,-2.55213045e-02,6.68262467e-02,-2.01556254e-02,2.78015584e-02,
9.82457027e-02,3.34636919e-04,2.17642933e-02,-5.99388257e-02,-2.10115552e-01,
-4.85517234e-02,-6.20813528e-03,1.15556993e-01,1.30376006e-02,-2.07846500e-02,
8.74024928e-02,-1.37679726e-01,-6.30606711e-02,1.40138909e-01,-5.44279739e-02,
4.75187972e-02,1.51111111e-01,1.19212925e-01,2.53212899e-02,-5.56647852e-02,
-7.22524673e-02,1.46540850e-01,-1.12158902e-01,3.41351852e-02,3.13940309e-02,
-7.21099367e-03,2.39809416e-02,-8.87056906e-03,-3.66651192e-02,-5.34733245e-03,
-1.57712738e-03,4.58260678e-04,5.34879193e-02,4.71186452e-02,-6.64952844e-02,
-5.30794784e-02,-1.82481617e-01,-8.64783302e-02,-1.57006935e-03,7.32607618e-02,
2.97255698e-03,5.18033504e-02,-8.61725211e-02,4.38375957e-02,3.60561907e-02,
9.80901420e-02,-6.02882728e-02,2.33838081e-01,9.39186290e-02,1.19099520e-01,
-4.71699014e-02,3.80691960e-02,7.19727054e-02,-2.14593947e-01,2.55344878e-03,
-8.61668810e-02,6.07249215e-02,1.21887244e-01,6.57463297e-02,3.57093965e-03,
-1.14405274e-01,1.66924626e-01,-6.42266078e-03,5.94464764e-02,-5.12439162e-02,
1.12532899e-01,2.07338184e-01,2.02794801e-02,-1.14427522e-01,1.59606516e-01,
-1.24006234e-01,-2.52350956e-01,-1.35339975e-01,3.04243471e-02,3.65376063e-02,
1.21672235e-01,-1.68160558e-01,-3.48808728e-02,-1.40739411e-01,-1.11449316e-01,
4.65482101e-02,4.56402227e-02,-1.08698145e-01,5.09435311e-02,-1.29653448e-02,
4.10828181e-02,9.57837924e-02,-3.05065662e-01,-1.49813499e-02,1.27456546e-01,
3.06429602e-02,2.38209933e-01,3.88150141e-02,-1.09666534e-01,-7.23160654e-02,
-4.90255468e-02,2.44353443e-01,1.23661687e-03,3.61634374e-01,6.94036186e-02,
6.34075105e-02,9.76807401e-02,-2.84946430e-02,-2.42669135e-01,1.79470390e-01,
4.00562525e-01,-1.22240819e-01,-1.27819687e-01,8.68403837e-02,5.77097991e-03,
7.96961263e-02,-1.15645893e-01,-6.80346191e-02,-3.53477709e-02,-6.44504651e-02,
-1.28682405e-01,2.59663705e-02,1.61504731e-01,1.56723440e-01,8.04094449e-02,
-1.16706686e-02,-3.07074152e-02,-2.51945350e-02,1.99156925e-02,1.11054264e-01,
-3.87260653e-02,1.58651173e-01,-7.15587363e-02,1.01111189e-01,4.83064121e-03,
5.42385615e-02,1.07534736e-01,-8.51183664e-03,6.40385598e-02,-3.79658788e-02,
2.35298816e-02,4.05187681e-02,-6.50688484e-02,-2.35719346e-02,-2.33854651e-01,
7.89673403e-02,-2.78775513e-01,-8.80126581e-02,1.52650461e-01,5.82733974e-02,
2.46598572e-02,1.25284731e-01,1.40375391e-01,3.78254265e-01,2.73162834e-02,
1.79085344e-01,-6.76245689e-02,-2.59549409e-01,-2.06115603e-01,-1.49967643e-02,
1.20434836e-01,2.28349105e-01,-1.00772925e-01,-1.03846163e-01,2.77775023e-02,
-1.77394360e-01,-8.97410288e-02,-5.75147060e-05,1.25284687e-01,1.28202755e-02,
3.53997469e-01,3.65800783e-02,7.03257844e-02,1.83061808e-01,2.11424530e-02,
-9.58927423e-02,9.15805474e-02,-1.95619226e-01,5.70643917e-02,3.84404249e-02,
-1.48078650e-01,4.16817725e-01,-1.74063761e-02,-5.87426722e-02,-1.02619559e-01,
-2.93521702e-01,3.80115271e-01,7.33325109e-02,-4.62963916e-02,-5.94569854e-02,
-3.62542272e-01,2.30949819e-02,-1.67667106e-01,-1.02089323e-01,8.88848826e-02,
4.49669501e-03,-1.13774903e-01,-3.50719988e-02,9.24067870e-02,1.05828634e-02,
1.64489280e-02,-1.12500936e-01,9.90881473e-02,5.39704025e-01,2.18990728e-01,
2.38976896e-01,-4.90889363e-02,3.06316502e-02,-1.84378266e-01,7.18371645e-02,
-2.54053503e-01,2.25560308e-01,4.68007624e-02,3.05778235e-02,-3.38826515e-03,
4.15766016e-02,4.86096069e-02,-1.70093983e-01,-1.14604689e-01,9.31595638e-03,
2.26216950e-02,-1.54290786e-02,3.77023555e-02,-1.20774001e-01,5.94094619e-02,
1.23752773e-01,1.69374391e-01,-8.06501135e-02,3.19265053e-02,-4.44423966e-02,
2.33673438e-01,-2.28808105e-01,-9.43957344e-02,2.83386528e-01,8.49251524e-02,
3.50535035e-01,4.20053631e-01,7.50783086e-02,5.84365129e-01,1.30978776e-02,
1.56238033e-02,-4.52799469e-01,1.59599856e-02,-7.39503503e-02,2.75402129e-01,
-1.24016762e-01,1.13426745e-01,-8.81593302e-03,-1.40548617e-01,1.49839461e-01,
2.60895817e-03,6.68359816e-01,-2.92061716e-01,-1.07475273e-01,1.43623605e-01,
1.71543062e-01,3.38880382e-02,2.76373960e-02,-2.93862913e-02,3.72718066e-01,
4.02590603e-01,8.04312900e-02,-7.99211189e-02,1.18750811e-01,-8.39421824e-02,
-6.53647780e-02,1.14252172e-01,6.25404596e-01,2.22816523e-02,-5.31733632e-02,
-8.26265737e-02,9.10401568e-02,-9.10636503e-03,5.69726480e-03,2.86346134e-02,
5.14024794e-02,-8.42284225e-03,7.40316063e-02,7.94686377e-03,-6.24915138e-02,
1.33094400e-01,-9.19386148e-02,-6.56237155e-02,-9.82331112e-02,-5.02516851e-02,
-8.18257183e-02,-6.64831102e-02,-4.40903008e-02,-5.14823049e-02,2.61124559e-02,
5.48633114e-02,-1.40629724e-01,-1.99524611e-01,2.35237032e-02,3.66102941e-02,
-1.38365149e-01,-1.25865452e-02,-7.52031431e-02,9.55156535e-02,3.34173068e-02,
1.42034903e-01,-1.35148153e-01,4.90437001e-02,-2.49404311e-02,1.20290350e-02,
8.75523165e-02,3.15532205e-03,1.03680007e-01,1.86124802e-01,-3.20929319e-01,
-5.44140413e-02,1.75536290e-01,-4.36837552e-03,1.52469635e-01,4.18647639e-02,
3.44903395e-02,1.10799372e-01,-2.45049149e-01,3.95298719e-01,-1.16363028e-03,
1.75885588e-01,-1.33816181e-02,-2.27063328e-01,7.21123591e-02,-3.55139468e-03,
-1.72587857e-01,1.70649424e-01,9.16539226e-03,2.42099866e-01,1.14830159e-01,
2.14380056e-01,-1.14536146e-02,-7.72681832e-02,-1.21501178e-01,-1.31148890e-01,
3.17303956e-01,1.53053045e-01,5.10781407e-02,-1.11694969e-02,4.47109938e-02,
-1.83667868e-01,-3.48145217e-02,-4.77108127e-03,-3.94449532e-02,1.48281995e-02,
2.14387879e-01,-1.83121057e-03,-2.96449095e-01,3.49388197e-02,-1.23501045e-03,
2.89902508e-01,-1.37612969e-01,3.73779014e-02,-1.29084751e-01,-1.58331469e-01,
-6.82069138e-02,-1.69050753e-01,-1.47140399e-01,-8.59659463e-02,-6.28026500e-02,
-2.11773276e-01,-1.24153553e-03,-2.30688691e-01,-1.96219116e-01,6.99160472e-02,
-5.75257614e-02,-1.17969863e-01,1.86818182e-01,6.14661984e-02,4.67154197e-02,
1.63966119e-01,2.28152121e-03,-6.61425367e-02,4.45464700e-02,1.01748437e-01,
2.89416965e-02,1.49787823e-02,-7.86132589e-02,-1.85987877e-03,9.38262194e-02,
-3.64674488e-03,4.01387624e-02,-5.68605475e-02,-1.34115860e-01,-3.69781554e-02,
2.97475606e-01,-4.28546816e-02,1.04271710e-01,-1.31019592e-01,-5.29792979e-02,
3.29381563e-02,-1.37506843e-01,-4.70434017e-02,-5.30824438e-03,-3.56988832e-02,
1.08852595e-01,-3.71346883e-02,2.92783938e-02,-1.24993399e-01,-7.56020751e-03,
-5.04123308e-02,-8.32932740e-02,-4.21686843e-02,-7.86448121e-02,7.17731640e-02,
-2.25551706e-02,9.86333787e-02,-5.26499934e-02,3.96328121e-02,2.23016024e-01,
-2.08614618e-01,4.86018397e-02,9.67139453e-02,-1.81890309e-01,-3.21199149e-02,
3.54960784e-02,2.34159261e-01,2.74566598e-02,-1.30201265e-01,1.21011406e-01,
1.30378902e-01,-1.34289458e-01,-4.94853258e-02,-9.33203287e-03,-2.09409237e-01,
2.78753936e-01,-2.73771621e-02,6.77389503e-02,-2.15276293e-02,6.31072968e-02,
1.25844879e-02,1.80945895e-03,1.47651628e-01,1.92327365e-01,3.16305697e-01,
-2.83690598e-02,-5.73992077e-03,1.18915085e-03,-2.17868779e-02,-5.63529262e-04,
-6.54942021e-02,-6.66250810e-02,4.40039858e-02,1.13554165e-01,2.15390921e-01,
1.52350456e-01,1.52536228e-01,-1.53619573e-02,1.69715881e-01,-1.35630310e-01,
6.81994632e-02,-5.20527847e-02,2.65155941e-01,-2.01587096e-01,9.53069255e-02,
-1.65526822e-01,-6.43487200e-02,4.76190038e-02,1.60136327e-01,6.96040168e-02,
-5.88197522e-02,7.63830245e-02,2.62701333e-01,2.68305372e-02,1.47242785e-01,
1.59857988e-01,7.31005520e-02,1.22009143e-01,5.81099093e-02,1.47970244e-01,
6.43070415e-02,-3.77103832e-04,8.22108909e-02,-2.21977253e-02,-1.92103624e-01,
-3.47370952e-01,-6.31673485e-02,7.47192428e-02,8.39377195e-02,-3.11435282e-01,
5.79330772e-02,-1.65911280e-02,1.65016696e-01,-2.87798882e-01,1.53815836e-01,
1.17728822e-01,-1.15889452e-01,2.43575811e-01,-1.85522527e-01,-2.17669845e-01,
-4.85363863e-02,-3.23162079e-01,6.52481019e-02,-2.42941957e-02,-1.90450579e-01,
3.96809950e-02,-3.99391018e-02,-2.81773537e-01,-3.33710074e-01,-3.40135902e-01,
-4.09819424e-01,4.43179868e-02,-3.31814200e-01,5.42795360e-02,-8.62325057e-02,
-4.99535687e-02,-3.03651929e-01,-2.93360315e-02,-2.76655406e-01,-2.85929680e-01,
-1.32556826e-01,3.01478028e-01,6.68417066e-02,-5.24568197e-04,-1.34572178e-01,
-3.45220044e-02,2.23329980e-02,1.83223173e-01,-1.82154655e-01,-8.19223076e-02,
1.41936898e-01,-9.66066346e-02,-8.40985924e-02,-1.70723975e-01,-2.64420390e-01,
-7.91029036e-02,6.97386637e-02,1.35506302e-01,3.87456238e-01,-1.72875047e-01,
-1.90760940e-01,-5.90401888e-01,-4.77225125e-01,-2.79302448e-01,9.67117548e-02,
-2.35770568e-01,-2.28259519e-01,4.67108190e-02,-3.83184701e-01,2.36348882e-02,
3.14821750e-01,3.81960541e-01,-8.61689895e-02,-1.44099712e-01,-1.95875764e-01,
1.93146527e-01,-1.65192351e-01,-1.66691542e-01,-7.65915439e-02,4.12256308e-02,
1.26696467e-01,-1.19167529e-01,1.59478903e-01,-2.29734644e-01,-6.01739772e-02,
-6.34635240e-02,-1.20285945e-02,-2.08484814e-01,6.02759654e-03,-6.14271872e-03,
-1.25546843e-01,1.36449024e-01,3.01163699e-02,1.58813938e-01,3.24901789e-02,
9.20997858e-02,2.02805132e-01,1.33474961e-01,4.15750705e-02,-2.19849974e-01,
9.37085077e-02,-2.00009137e-01,1.45917684e-02,4.95668352e-02,2.55726904e-01,
-1.95890740e-01,-1.30846664e-01,1.21722721e-01,-1.57884713e-02,-8.69543552e-02,
-1.70056865e-01,8.50659236e-02,1.66588932e-01,2.28689946e-02,-7.21833333e-02,
5.70662916e-02,-3.59536707e-02,-8.28118324e-02,-7.41208345e-02,-4.55507785e-02,
8.57198164e-02,5.04340194e-02,-7.50177279e-02,-2.13433251e-01,-2.85771792e-03,
1.64288208e-01,1.30025595e-01,2.25031376e-02,-4.78798822e-02,1.62311837e-01,
-1.66864142e-01,8.04249346e-02,-2.03709498e-01,-7.48680322e-04,4.00364846e-02,
-4.93531339e-02,6.35598242e-01,2.84413189e-01,-1.68291196e-01,-2.19518170e-02,
-6.54068589e-01,4.51047011e-02,-2.94248350e-02,1.98100563e-02,2.76522100e-01,
-3.29036593e-01,-1.42309561e-01,-8.11485108e-03,-2.35098481e-01,1.16752245e-01,
1.25292987e-01,1.24619506e-01,5.65051436e-01,-4.80767302e-02,1.69722363e-02,
-4.72442918e-02,-4.12414402e-01,-1.33771211e-01,6.14003390e-02,-9.72108394e-02,
-1.25401124e-01,1.52997717e-01,3.64512205e-02,-8.24803039e-02,-7.73525238e-02,
-4.49735299e-02,7.61345848e-02,-3.29220220e-02,-3.22739363e-01,6.33539632e-02,
-8.83232728e-02,6.59680590e-02,8.16150848e-03,2.54639070e-02,7.24617913e-02,
5.13760149e-02,-1.58361539e-01,-7.83724189e-02,-1.01004712e-01,1.08154339e-03,
1.69220101e-02,-3.29234302e-02,1.41428530e-01,-9.96551886e-02,6.33999258e-02,
-3.60093564e-01,-2.36047897e-02,1.03522316e-02,1.18166305e-01,-2.65634328e-01,
-1.29722282e-01,-3.14025283e-02,-5.86904362e-02,-3.49119663e-01,4.32982981e-01,
-8.90396461e-02,-1.00077361e-01,2.44446266e-02,-3.33469361e-02,2.67410390e-02,
-6.57808930e-02,6.82637691e-02,-9.60899144e-02,-3.25781591e-02,-5.57306036e-02,
-1.03795625e-01,1.16226906e-02,7.19653210e-03,-1.86785832e-01,-8.02869536e-03,
1.02850780e-01,2.72214767e-02,5.02057299e-02,-1.51842525e-02,1.22402631e-01,
1.16985522e-01,1.91081405e-01,-3.84058058e-02,1.12347370e-02,-8.78740922e-02,
-9.11172256e-02,-2.07709134e-01,1.01550713e-01,-2.20352232e-01,-2.42523104e-02,
-1.06286056e-01,8.06475356e-02,-1.12959985e-02,-1.97631642e-02,-6.92336038e-02,
2.08350137e-01,-6.35019243e-02,4.89012664e-03,1.08412668e-01,2.72555538e-02,
1.01033170e-02,1.01402655e-01,2.88213324e-02,-3.50362927e-01,2.07765579e-01,
-1.05602548e-01,-1.45818666e-01,-4.08105850e-01,-4.46168840e-01,9.32929292e-02,
2.32275039e-01,-4.32221629e-02,6.06376491e-02,2.89846957e-01,-2.63430569e-02,
3.81512046e-02,-6.30514175e-02,8.42199661e-03,-4.84880917e-02,6.53081462e-02,
-2.60131806e-01,1.13507986e-01,-4.85388599e-02,-2.36444533e-01,-6.47124052e-01,
-1.53161794e-01,-1.88023165e-01,1.65276080e-02,-6.43587038e-02,-1.17210820e-01,
9.35075358e-02,2.84529440e-02,-2.75446892e-01,4.66492747e-05,-1.36533126e-01,
-2.06028059e-01,-1.35565162e-01,-1.08188793e-01,-3.65040116e-02,7.52729923e-02,
-1.10346153e-01,3.42205882e-01,-3.52194846e-01,-1.98191628e-02,1.32186577e-01,
-2.51148015e-01,-1.85549691e-01,4.48794961e-02,-9.35998037e-02,-2.08025038e-01,
-2.27948099e-01,-1.02316491e-01,1.32618710e-01,-6.61420003e-02,-2.17785612e-01,
1.85017358e-03,6.11306876e-02,-1.13412149e-01,-1.57772761e-03,-2.01084509e-01,
-2.03664511e-01,-4.42086071e-01,-2.68429607e-01,-9.38094407e-02,-2.38219410e-01,
-5.21019548e-02,-1.76619217e-01,1.97742209e-01,-9.04317200e-02,5.61824478e-02,
-3.94664317e-01,1.48167074e-01,3.57975438e-02,-2.71984059e-02,-2.11081937e-01,
-3.49937141e-01,9.97512192e-02,9.32046399e-02,-1.53796375e-01,1.92228913e-01,
9.24475864e-03,-1.63600013e-01,-1.52414918e-01,-6.00491054e-02,-4.30498198e-02,
-5.85022056e-03,-2.73107439e-01,-8.32059234e-02,-3.87473814e-02,-3.52433950e-01,
4.21305507e-01,-7.77814612e-02,1.83324605e-01,-5.09088039e-02,8.19318146e-02,
-9.12891980e-03,1.04847243e-02,5.34900501e-02,-2.62946814e-01,1.32802455e-02,
8.12330768e-02,-2.41612315e-01,1.86823122e-02,-1.58081651e-02,4.80221212e-02,
8.49075392e-02,1.66077822e-01,-1.94724604e-01,8.94766003e-02,9.99324992e-02,
3.24519794e-03,7.80948475e-02,-5.50391227e-02,-3.62036824e-02,-1.31412437e-02,
-1.16628885e-01,1.56317409e-02,5.72948568e-02,2.54478693e-01,-9.31788782e-06,
-1.79775823e-02,1.89573262e-02,6.57942146e-02,1.12825066e-01,-9.15801600e-02,
-6.74871122e-03,4.28642221e-02,-4.13770191e-02,4.17558104e-02,3.80592085e-02,
-1.33742079e-01,1.73675455e-02,2.72762757e-02,7.14615658e-02,1.43177629e-01,
7.57764280e-02,-7.46237710e-02,7.93256387e-02,-1.14192456e-01,8.14751815e-03,
-5.32852709e-01,5.69326766e-02,6.95512518e-02,-1.80854693e-01,-2.23704711e-01,
9.24613178e-02,-1.70990840e-01,1.30422086e-01,-7.33378530e-01,-6.04074374e-02,
-1.03131384e-02,-4.93648127e-02,-1.74312532e-01,1.58757493e-01,-9.24875028e-03,
1.11182377e-01,-2.67787784e-01,3.44274677e-02,9.04941484e-02,-2.13704947e-02,
8.50490108e-03,-1.00202188e-01,-1.19304776e-01,-1.82137325e-01,7.81492591e-02,
-1.41646370e-01,2.09279373e-01,-2.66374126e-02,-2.34056413e-01,1.71139706e-02,
2.53072411e-01,-2.93587092e-02,1.70555606e-01,3.26630548e-02,1.93422660e-01,
3.46585028e-02,1.05972297e-01,7.61661381e-02,-2.94560790e-01,5.36891930e-02,
-3.99620719e-02,4.62345220e-03,4.86223884e-02,-1.98693946e-03,1.32923163e-02,
-8.72826576e-02,2.37127453e-01,-1.42023578e-01,5.89277558e-02,-1.88282385e-01,
1.94096178e-01,-7.13402554e-02,2.27588296e-01,-1.90630302e-01,-1.24289028e-01,
4.27725315e-02,2.25308202e-02,6.96431473e-02,4.23467606e-01,-5.25122762e-01,
1.95230886e-01,1.22975320e-01,-1.41268224e-01,-1.47801101e-01,1.22240214e-02,
1.05891146e-01,-4.19930637e-01,-1.85515270e-01,3.44941057e-02,-1.82644725e-01,
-3.25130224e-01,3.12645495e-01,-1.78185105e-01,-6.56468198e-02,-4.50167924e-01,
3.29840630e-01,-4.39997576e-02,-9.26471874e-02,-2.93507501e-02,1.35565192e-01,
-4.97672409e-02,1.43987656e-01,-3.12740393e-02,1.61096424e-01,-6.96033463e-02,
-4.21638750e-02,5.81935287e-01,-4.04732116e-02,1.08439021e-01,5.87598443e-01,
4.42235529e-01,-7.15716369e-03,1.88671812e-01,-5.05139828e-02,-3.51149850e-02,
-7.02241659e-02,-8.28715786e-02,-1.59292862e-01,-1.29961944e-03,1.02810115e-01,
1.57801837e-01,-1.21893827e-02,-3.52045268e-01,-1.26856923e-01,-2.65467435e-01,
-1.61318891e-02,-9.45757106e-02,-2.15741143e-01,-1.39561379e-02,6.26696944e-02,
1.92627590e-02,-4.29419309e-01,-4.33728052e-03,9.98352841e-02,-3.57559845e-02,
1.43899191e-02,1.88226491e-01,-9.83076394e-02,-3.65389660e-02,2.15233058e-01,
-2.39922795e-02,6.70159832e-02,3.10280062e-02,1.50789330e-02,-5.10315001e-02,
2.34284680e-02,2.36039497e-02,1.89795107e-01,1.46029413e-01,9.74912196e-02,
-1.39495153e-02,-4.79174145e-02,-1.52584225e-01,5.33146746e-02,-7.08003342e-02,
-3.60552128e-03,7.26139825e-03,-9.55167264e-02,-1.41817599e-03,-7.47498795e-02,
-4.91009839e-02,-5.10960631e-02,-3.87052372e-02,3.51341162e-03,-8.64336491e-02,
7.23861381e-02,1.24825872e-01,1.05850371e-02,5.36484569e-02,-1.25370532e-01,
-4.35897075e-02,-4.59453277e-02,-4.38664295e-02,-4.33135740e-02,2.11649448e-01,
8.94938335e-02,8.81619081e-02,-1.44238189e-01,3.63479674e-01,-1.04055934e-01,
-2.74682455e-02,-8.08046535e-02,2.30474621e-01,-5.50396815e-02,1.56575650e-01,
5.33745766e-01,-1.24560595e-02,-7.40724280e-02,2.79754668e-01,3.36474814e-02,
-2.99789552e-02,1.48843810e-01,-1.90345403e-02,-6.49751276e-02,-2.32177526e-01,
1.51162148e-02,-1.23406380e-01,1.26725242e-01,6.94274008e-02,1.68511774e-02,
-5.86014532e-04,5.06471284e-02,8.50467756e-02,1.56432286e-01,-1.08825304e-01,
-9.89879109e-03,-4.35168266e-01,4.42187972e-02,4.85342089e-03,-7.73274675e-02,
2.66883969e-01,8.39575306e-02,-2.31052250e-01,9.38158203e-03,7.80096427e-02,
-2.38046855e-01,2.77922908e-03,-2.92562485e-01,1.29900470e-01,-1.58527225e-01,
-3.26743461e-02,5.46787716e-02,2.94452589e-02,-2.74833888e-01,-2.04564840e-01,
-7.14980364e-02,-4.39668540e-03,-2.72194862e-01,8.14553499e-02,7.94470981e-02,
-2.27229059e-01,-3.20073813e-01,-3.41738015e-02,-2.05161467e-01,1.11125663e-01,
-8.05385262e-02,-4.96672466e-02,1.97873175e-01,2.85944752e-02,-2.51656026e-01,
-1.29625229e-02,2.00580999e-01,-1.94064766e-01,2.30573006e-02,-4.93065864e-02,
-8.95360336e-02,-2.71032676e-02,-6.58866465e-02,-5.21433689e-02,-4.84572388e-02,
-3.06132175e-02,-1.01680316e-01,-4.79712225e-02,1.70481415e-03,3.92556712e-02,
1.76270567e-02,8.04991424e-02,-1.65662810e-01,3.85049134e-02,2.23080441e-02,
-1.07245713e-01,2.38400504e-01,-3.69202346e-02,-5.44441603e-02,1.65392444e-01,
-1.13849357e-01,-2.12022975e-01,1.35862872e-01,-8.48219097e-02,5.26060648e-02,
-1.36679932e-01,1.43145129e-01,1.51718035e-01,-1.30749255e-01,1.55669630e-01,
1.71087414e-01,-2.81895492e-02,6.88415170e-02,3.77771910e-03,3.42948511e-02,
1.99628994e-01,-7.61424378e-02,8.32307041e-02,-6.26682416e-02,3.46048102e-02,
5.72131760e-02,-1.39834970e-01,-7.00338110e-02,8.15590173e-02,-4.12641354e-02,
-1.52316138e-01,1.05019003e-01,1.64915055e-01,1.36645511e-01,7.96947852e-02,
3.54914479e-02,1.10118881e-01,6.39126962e-03,-4.29123975e-02,5.07885665e-02,
4.59268084e-03,-1.00130200e-01,4.27841842e-02,-1.00589544e-01,6.13230541e-02,
-2.07097158e-01,-1.13293901e-02,3.78880799e-02,-1.57864094e-02,1.00792237e-01,
-9.79453251e-02,-5.14504686e-02,-1.22081325e-03,-7.42426142e-02,2.06187032e-02,
-2.52963766e-03,-8.98665190e-02,7.97689930e-02,2.20166534e-01,2.98246831e-01,
4.56328280e-02,8.43086243e-02,-1.30617753e-01,7.64064938e-02,-4.79001962e-02,
4.01025802e-01,-1.46494031e-01,-1.68552682e-01,6.12978749e-02,1.49648059e-02,
4.40398641e-02,9.30835381e-02,-1.02697462e-01,1.11752050e-02,-1.53431132e-01,
3.06061357e-01,-1.05250128e-01,5.25643714e-02,-1.91440254e-01,1.07967183e-01,
-3.29077169e-02,-3.11898571e-02,2.86886752e-01,-1.72987521e-01,-7.09668472e-02,
3.79075715e-03,1.07956618e-01,-9.11465809e-02,-7.32443482e-02,-3.03996336e-02,
-7.72381350e-02,-1.29283980e-01,4.59074602e-02,-9.43932310e-02,-2.49033064e-01,
1.51515394e-01,-1.92077711e-01,-5.74669950e-02,1.09361276e-01,2.86718160e-01,
-1.57436490e-01,4.82099839e-02,-1.87994763e-01,-3.61579418e-01,-1.18845560e-01,
1.64659813e-01,1.27987256e-02,2.09218338e-01,-4.53111865e-02,-3.97800840e-03,
-1.32219940e-01,-1.37595654e-01,1.83272250e-02,4.20424789e-01,6.04339726e-02,
7.38315582e-02,1.07431285e-01,-3.34135927e-02,2.70966031e-02,-7.36708865e-02,
-4.75715511e-02,5.35013080e-02,-6.98385164e-02,-3.13921124e-02,-1.03364132e-01,
-3.86623070e-02,4.85544577e-02,-1.79337740e-01,-4.93788458e-02,-5.17251790e-02,
4.25648578e-02,-1.01240791e-01,3.51868458e-02,9.65040699e-02,-1.60736412e-01,
1.87237158e-01,4.94965294e-04,5.64546846e-02,4.29985765e-03,-6.48947656e-02,
-3.32312658e-03,8.58680457e-02,-1.32859889e-02,-1.46723673e-01,-3.22294123e-02,
-3.96698341e-02,1.49844497e-01,3.75991315e-02,1.53343454e-01,2.17218399e-01,
-2.05530673e-01,-1.25629112e-01,2.01507896e-01,1.49046555e-01,1.58582538e-01,
-1.07906245e-01,1.30987257e-01,4.04181302e-01,2.18835339e-01,-8.52292869e-03,
-5.92775010e-02,2.22880632e-01,-2.07706556e-01,-6.45352378e-02,-5.01305573e-02,
1.42019600e-01,-3.40889916e-02,4.18691114e-02,-1.24465853e-01,-2.90045321e-01,
4.66762669e-02,-9.86589398e-03,-1.39932185e-01,9.73444059e-02,5.63401496e-04,
2.22554684e-01,-9.69676599e-02,-3.43740344e-01,-8.75610188e-02,4.49878722e-02,
1.17467038e-01,1.07547224e-01,-6.78994954e-02,2.68328488e-01,-2.55415678e-01,
1.53088376e-01,-7.44614601e-02,-6.79980814e-02,4.97249439e-02,-3.69298086e-02,
1.16413511e-01,7.29598850e-02,-1.09871939e-01,-2.57814564e-02,-1.10946126e-01,
2.02641189e-01,-5.94786480e-02,2.03861445e-01,1.15329005e-01,1.27804339e-01,
6.42529801e-02,1.30665377e-01,3.69401015e-02,-5.06716706e-02,-2.65162140e-01,
3.74497660e-02,3.05655420e-01,4.62824926e-02,4.54519503e-02,1.69729024e-01,
-4.35858458e-01,-1.07978918e-01,6.07105941e-02,9.11807939e-02,8.12886506e-02,
-7.93156326e-02,6.04372062e-02,3.86806391e-02,-2.68788993e-01,4.59445626e-01,
1.04619544e-02,-5.45931570e-02,3.68256390e-01,1.27690598e-01,-1.91914607e-02,
1.28213972e-01,2.54779100e-01,1.71164185e-01,2.60219306e-01,-3.95642482e-02,
4.97774541e-01,2.66729504e-01,-9.90784820e-03,4.26753424e-03,7.66793072e-01,
1.56465277e-01,-4.87907566e-02,-5.40001728e-02,6.59836531e-01,4.00226682e-01,
2.73190737e-01,-2.24173695e-01,7.03118861e-01,-1.05989516e-01,9.66703519e-03,
9.11942795e-02,-1.31003130e-02,-1.53761804e-01,-5.51563501e-02,-2.04292893e-01,
-2.10754909e-02,-1.85436308e-01,-1.81559157e-02,-1.39579296e-01,2.21363343e-02,
4.71130669e-01,-3.81490320e-01,9.60549191e-02,2.41233874e-02,1.26857474e-01,
-3.34086306e-02,-2.22010780e-02,1.37736946e-01,6.51961982e-01,-1.38579816e-01,
1.11711092e-01,5.02209738e-03,3.33432518e-02,2.07541898e-01,-9.26657692e-02,
-2.11968079e-01,6.12217128e-01,5.78441434e-02,-1.87255740e-01,8.50765258e-02,
3.52837801e-01,-2.36148816e-02,1.08601116e-01,1.39031485e-01,2.10983559e-01,
-1.18577234e-01,-4.54597920e-02,2.27192864e-02,2.38647894e-03,-7.77261853e-02,
-1.78386495e-01,-2.27989897e-01,1.53715108e-02,7.67351538e-02,-3.90556790e-02,
-1.57149330e-01,-4.60281633e-02,-1.97291479e-01,-9.65324510e-03,5.62095270e-02,
-2.34619458e-03,-2.89792567e-01,6.24390393e-02,-4.50234078e-02,1.15223043e-01,
-1.89098436e-02,1.14521205e-01,7.50792176e-02,-6.59093931e-02,2.02354759e-01,
-1.66667178e-01,-8.32938477e-02,-7.19449520e-02,-1.97500572e-01,1.21906310e-01,
-6.91373870e-02,-3.25312885e-03,-1.13581464e-01,1.20254479e-01,2.04057679e-01,
2.08751485e-02,3.40354815e-02,1.56560261e-02,-4.33142520e-02,-2.04566233e-02,
-1.13941826e-01,-4.07658756e-01,1.93758190e-01,8.71244818e-02,-1.14825226e-01,
-1.58711225e-01,1.32883349e-02,6.85027987e-02,-1.44118950e-01,-5.97394891e-02,
-2.46906597e-02,1.09041736e-01,9.02974531e-02,1.39249876e-01,-3.70033719e-02,
-1.54410638e-02,-2.48266608e-02,7.44209215e-02,-1.89581126e-01,-1.72835574e-01,
8.78153276e-03,-1.58347175e-01,-4.55997258e-01,-2.42654197e-02,6.77298829e-02,
-7.23819714e-03,3.99651155e-02,-3.20672952e-02,2.25574616e-02,8.52996334e-02,
-1.01731613e-01,-2.16406748e-01,-1.23424158e-01,-1.33582875e-01,3.87004158e-03,
-1.66469306e-01,1.83600739e-01,-6.18173815e-02,-1.17154323e-01,-2.75045224e-02,
-1.54712111e-01,2.70127449e-02,-8.89328793e-02,1.56186387e-01,3.92935351e-02,
-1.06726266e-01,-4.08575274e-02,-2.31901333e-01,-1.16577540e-02,-1.77114949e-01,
-1.33389048e-03,-5.20183332e-02,1.42999156e-03,5.15654497e-02,6.95902109e-02,
1.18549831e-01,2.52945051e-02,4.11061523e-03,8.71002674e-02,-2.53027324e-02,
-2.91847009e-02,2.73005906e-02,-2.14319257e-03,-8.15744773e-02,1.74164139e-02,
1.03570379e-01,8.08494240e-02,-2.41544819e-03,-1.19494282e-01,1.19091071e-01,
1.41493633e-01,4.19856422e-02,6.08462431e-02,1.18981712e-01,7.32039362e-02,
1.02907963e-01,-2.41108192e-03,6.56680763e-02,1.66500807e-02,-1.07449386e-02,
8.35994333e-02,-1.93814665e-01,-3.09881121e-02,-7.45611042e-02,-2.86296066e-02,
-3.57266795e-03,-4.90835682e-02,-7.76702836e-02,-3.58262546e-02,1.18645672e-02,
5.40031195e-02,-5.64243905e-02,2.01019824e-01,2.13147640e-01,-1.80600464e-01,
-6.87382976e-03,-7.26663321e-02,-2.87321657e-01,8.44292268e-02,-2.98132837e-01,
1.86866701e-01,-7.69207999e-02,9.32715461e-02,-2.05657855e-01,-1.70253903e-01,
-2.78086960e-01,-1.47349566e-01,6.25767708e-02,-2.34381884e-01,2.38829523e-01,
1.35888085e-01,-6.52682334e-02,1.88760310e-02,-2.37081304e-01,3.50875258e-02,
-4.13502231e-02,-2.92352727e-03,4.37568426e-02,6.78546578e-02,1.42701864e-01,
-5.79607561e-02,-1.32924899e-01,6.99279457e-02,-3.45859230e-02,-1.75438821e-01,
1.95194528e-01,-2.15593517e-01,1.70289949e-01,-2.13524804e-01,1.81073442e-01,
-4.14359942e-02,6.38822988e-02,-2.55583942e-01,-1.96309775e-01,-9.79906414e-04,
5.82119785e-02,-5.43086752e-02,-8.25328454e-02,-5.40567786e-02,-8.36055633e-03,
-6.89008236e-02,6.45729080e-02,1.11998275e-01,-6.76218569e-02,1.88517168e-01,
7.51659721e-02,-1.72769085e-01,3.32347490e-03,-4.24178988e-02,1.03310287e-01,
-1.97774157e-01,-1.11723401e-01,-5.89340925e-03,1.20871097e-01,-1.15183733e-01,
-1.90044604e-02,2.69176930e-01,-2.80843526e-02,9.43582729e-02,-2.70787507e-01,
3.38050514e-03,1.28562991e-02,2.54726022e-01,1.19008221e-01,-3.34705934e-02,
8.52776095e-02,-2.02556536e-01,-3.63264792e-02,-3.39015692e-01,6.58158809e-02,
-1.32106215e-01,-4.82744038e-01,-1.50402799e-01,-8.66359994e-02,-4.13213879e-01,
4.28786539e-02,6.05568141e-02,5.81321586e-03,-3.16190720e-01,-2.84737676e-01,
6.19637743e-02,9.44140851e-02,-2.76436687e-01,-1.05332322e-01,-8.84020701e-02,
3.28338116e-01,-1.43538088e-01,1.33817764e-02,3.54233161e-02,-2.10634992e-01,
-2.26136759e-01,1.94759265e-01,4.01511528e-02,-4.60123777e-01,-4.53116089e-01,
-6.58191562e-01,-6.72327161e-01,-3.74300368e-02,-5.11200786e-01,-4.67626005e-01,
-1.79153427e-01,-5.28754711e-01,-1.77781761e-01,4.06277888e-02,-4.27255124e-01,
-1.69266433e-01,1.25418067e-01,-6.00086451e-01,-4.47720885e-01,1.69872731e-01,
-5.77296168e-02,-1.45797849e-01,-6.14158928e-01,3.80206592e-02,-2.40579359e-02,
-7.03817725e-01,-5.56002796e-01,-1.16353858e+00,-3.11974734e-01,-1.12723947e-01,
-3.38119179e-01,-1.65806219e-01,-2.37082839e-01,-1.59636363e-01,-2.58510951e-02,
-7.63921589e-02,-1.41537681e-01,-1.26748249e-01,1.44217402e-01,1.57428935e-01,
-4.67155091e-02,8.93110409e-02,1.00365959e-01,-1.96140751e-01,-4.19292934e-02,
-4.40610588e-01,4.97843653e-01,-2.62348831e-01,6.12793155e-02,-1.81162301e-02,
5.17261326e-01,1.53721916e-02,-1.29691288e-01,-1.88997760e-01,3.02972734e-01,
-2.69760638e-01,-7.33435387e-03,3.13894093e-01,2.43770346e-01,2.93751121e-01,
4.08486485e-01,9.04764514e-03,2.50621811e-02,5.39393425e-02,2.58962926e-03,
-1.18247375e-01,3.14156443e-01,2.99593434e-03,2.33457416e-01,-1.33797526e-01,
-6.73401654e-02,3.57862487e-02,1.67592466e-02,-7.42822066e-02,-9.87934694e-02,
2.74387598e-01,-2.56126463e-01,-1.21804096e-01,-1.01076625e-01,8.24366659e-02,
-1.95165977e-01,1.74416751e-01,1.33255467e-01,5.45554906e-02,2.54028998e-02,
-9.19373557e-02,5.42856418e-02,-5.96830063e-02,-5.67258224e-02,-4.04430144e-02,
1.40338808e-01,4.77424502e-01,1.17084652e-01,1.98260516e-01,-9.38881487e-02,
2.75525004e-01,-7.04756081e-02,1.47127867e-01,3.26474048e-02,-3.17632519e-02,
2.15512559e-01,2.87715137e-01,-3.79029989e-01,-4.38139528e-01,-3.22689593e-01,
-1.09995857e-01,6.12829998e-03,2.31912598e-01,1.83058754e-01,4.31839019e-01,
-4.49106872e-01,-5.28403103e-01,-7.42490515e-02,-1.50093302e-01,3.93763006e-01,
7.18208626e-02,-1.50562540e-01,-7.90051818e-01,-7.74315596e-02,3.14294398e-02,
-2.98179090e-01,-2.11541727e-01,2.43056193e-02,1.02843888e-01,-4.70523298e-01,
-3.40506107e-01,1.08437829e-01,1.25085354e-01,-1.34214684e-01,-1.15672812e-01,
3.76330659e-04,-2.56344050e-01,-2.07038492e-01,1.30072027e-01,5.99000789e-02,
1.73592880e-01,8.85351747e-03,-3.12026385e-02,2.10929438e-01,-1.01951130e-01,
-1.20168336e-01,4.59523872e-03,7.12842569e-02,-3.63359094e-01,-1.83607623e-01,
4.84448284e-01,5.24280630e-02,2.36107320e-01,5.17734066e-02,9.76477563e-02,
1.67383105e-01,9.13096033e-03,2.79198755e-02,6.03011623e-03,1.20762631e-01,
-1.32403806e-01,1.75305530e-01,-3.33431423e-01,3.63121271e-01,2.24442512e-01,
8.10114443e-02,-2.22685397e-01,1.00125968e-01,3.76451425e-02,-1.06768839e-01,
-2.97464758e-01,-8.84612128e-02,3.67429033e-02,-2.64181173e-03,3.52801718e-02,
-5.90591021e-02,-8.26152936e-02,9.29250568e-02,-3.29076387e-02,-3.94449979e-02,
2.32254975e-02,7.29947239e-02,6.88165501e-02,-2.97414288e-02,8.60774219e-02,
-1.18602395e-01,1.20293107e-02,-2.07034387e-02,1.88991114e-01,-8.34737122e-02,
-2.35808417e-02,5.48394285e-02,1.38889804e-01,-9.00787115e-02,-5.32438830e-02,
-1.93975881e-01,-1.75996087e-02,1.39909342e-01,3.74782681e-02,1.80910788e-02,
1.32181153e-01,6.82787225e-02,6.82322681e-02,-2.70549152e-02,1.83116898e-01,
1.88117728e-01,-9.02628526e-02,-3.23026508e-01,-9.72962528e-02,-2.07428392e-02,
-1.08286716e-01,-1.36958361e-01,-2.22011432e-01,7.37772435e-02,-9.08085853e-02,
-9.16910768e-02,1.46094978e-01,-7.63776302e-02,1.09946460e-01,-6.94751218e-02,
-1.72864452e-01,3.78265709e-01,2.26113833e-02,-2.48976171e-01,-4.14116412e-01,
-2.24235319e-02,-1.35285214e-01,1.18715711e-01,-7.38307536e-01,-1.92963451e-01,
-1.55353760e-02,-3.41956139e-01,-5.64652562e-01,-3.83947492e-02,5.29117556e-03,
-7.57622302e-01,-1.52398512e-01,1.82179704e-01,1.62079886e-01,1.10689159e-02,
-2.48591214e-01,-2.11206049e-01,-4.80269462e-01,-4.45808470e-02,-2.39021644e-01,
1.87907547e-01,-9.38895196e-02,-3.50076854e-02,1.16602875e-01,-2.26311475e-01,
2.32930616e-01,-7.58126155e-02,-7.54664168e-02,5.81875406e-02,-2.03650549e-01,
7.22918659e-02,-1.12797663e-01,-2.64958262e-01,4.81808931e-01,2.57281121e-02,
-2.02093422e-02,2.13202164e-02,-7.72916004e-02,-4.23282832e-01,-1.82244152e-01,
3.50824110e-02,2.25991890e-01,-1.97804254e-02,9.20178220e-02,1.22900985e-01,
2.97949344e-01,-4.45496589e-01,-5.34011841e-01,-7.81247079e-01,-2.17390776e-01,
-4.44952726e-01,-1.40373960e-01,-1.92829758e-01,3.28763425e-02,5.31370426e-03,
-2.48489127e-01,-3.10580194e-01,-2.06439599e-01,6.55659437e-01,-7.28376329e-01,
2.21936051e-02,-4.24591631e-01,-3.03879917e-01,-5.38053736e-03,-1.05591323e-02,
-5.10297477e-01,8.12529698e-02,-5.81127144e-02,-3.53248328e-01,-2.02189937e-01,
-3.64451498e-01,-3.75202745e-01,-1.88178159e-02,-4.37326908e-01,-1.14988379e-01,
8.08238517e-04,-4.72089499e-01,-5.60772717e-01,3.45029980e-01,3.53730768e-01,
-4.31766957e-01,-1.41278744e-01,-3.55666399e-01,-1.17269479e-01,-9.55876261e-02,
9.65487882e-02,-1.35707200e-01,3.62281919e-01,1.02770440e-01,8.73213187e-02,
-2.86766857e-01,3.09050977e-01,-1.43187925e-01,1.89810887e-01,-6.56453893e-02,
1.21244170e-01,-5.44148758e-02,2.15722606e-01,-1.74801767e-01,1.93894506e-01,
-1.32163331e-01,-1.61989212e-01,-3.36336821e-01,-8.24146688e-01,-7.99492002e-01,
-1.71106040e-01,-1.93264171e-01,1.38189554e-01,-1.14599131e-01,-2.09071785e-01,
-4.61843163e-01,-4.82015789e-01,3.56118888e-01,-8.62742215e-02,2.27696240e-01,
-2.13499352e-01,5.08128740e-02,2.11950913e-01,3.41350406e-01,-3.33172441e-01,
2.78344244e-01,-3.11500788e-01,-5.29144295e-02,-2.85070747e-01,-2.38411129e-02,
-1.37003213e-01,-1.31535739e-01,-4.23187017e-01,-8.12556446e-02,9.49917063e-02,
2.49800794e-02,-4.18090895e-02,1.95661515e-01,7.04045175e-04,-8.40141103e-02,
-2.86412477e-01,8.13944731e-03,2.91820496e-01,-2.43440613e-01,-2.92225808e-01,
3.17774306e-04,1.19366132e-01,-2.18579382e-01,-2.96525620e-02,-1.12633325e-01,
6.49354383e-02,-1.01903923e-01,1.31208897e-01,3.13921124e-01,3.56631368e-01,
7.10925087e-02,2.21929982e-01,9.47608575e-02,-9.20170024e-02,-1.18949048e-01,
-2.39721283e-01,1.95712671e-01,-2.68160611e-01,1.04326323e-01,-9.03607458e-02,
3.72969776e-01,2.32601032e-01,1.15375236e-01,1.66799631e-02,-4.98940945e-02,
5.60899824e-03,-6.73001036e-02,1.27993703e-01,5.90204634e-02,3.10355455e-01,
3.43909144e-01,3.36642444e-01,-3.04949284e-03,-2.82721221e-02,5.40897027e-02,
-5.71744181e-02,-1.07381515e-01,2.87757277e-01,4.72524241e-02,1.22056521e-01,
2.26817325e-01,-1.64682508e-01,1.16433926e-01,-5.96188568e-02,2.72191286e-01,
5.20828307e-01,-1.29892433e+00,-1.63770065e-01,2.57803172e-01,-3.49572390e-01,
2.82780230e-01,4.88328040e-02,-3.17352451e-02,-8.43022987e-02,8.72923493e-01,
2.14327902e-01,-2.45473962e-02,5.20661771e-01,3.67857397e-01,-5.55896908e-02,
2.94459581e-01,3.50199342e-01,2.31158528e-02,8.71009752e-02,-1.48043036e-01,
3.79967630e-01,2.67108470e-01,-1.20162003e-01,5.65261006e-01,6.58452809e-01,
-1.47262752e-01,-8.34795460e-02,5.71598671e-02,4.55425195e-02,4.47771046e-03,
-1.48274317e-01,-3.81245129e-02,1.12070672e-01,3.24055105e-01,2.26632863e-01,
4.15099040e-02,-3.12271148e-01,-1.00258570e-02,-3.74956191e-01,-1.15778908e-01,
-8.33404735e-02,-3.47419441e-01,2.58140862e-01,4.15308587e-02,1.06423631e-01,
-4.32132155e-01,6.31110370e-02,-9.37047526e-02,-2.63359457e-01,-9.73692983e-02,
4.06020701e-01,6.82955623e-01,2.97041368e-02,-5.38911879e-01,2.65556157e-01,
-1.62839577e-01,-2.53883481e-01,-1.04139201e-01,-4.49963734e-02,1.15770809e-02,
1.28463041e-02,-1.21629380e-01,-3.86720337e-02,1.00496225e-01,1.29203841e-01,
-6.56156987e-02,-1.59281313e-01,5.40960245e-02,-8.12404510e-03,4.91909236e-02,
6.52537495e-02,-5.68527728e-02,-1.30838202e-02,8.80251080e-02,3.48548310e-05,
-3.24484594e-02,-7.32062012e-02,-3.83989960e-02,8.57348274e-03,-1.49795618e-02,
1.31587870e-02,-1.13193497e-01,-1.02490999e-01,-2.76189018e-02,3.28509063e-02,
1.10376760e-01,-1.50796577e-01,-2.69829035e-02,6.52909428e-02,6.24114238e-02,
-9.53998193e-02,-8.49285126e-02,2.55038962e-02,2.29999557e-01,2.60792911e-01,
-7.75389150e-02,-2.37505529e-02,9.23032612e-02,-1.44129351e-01,3.63373101e-01,
-4.53648530e-02,-2.77681708e-01,4.80779037e-02,1.10128999e-01,-3.42749581e-02,
1.24267988e-01,-2.86935512e-02,2.45413914e-01,-6.85697049e-02,-8.87727588e-02,
-1.43569052e-01,2.84936931e-03,-3.29132080e-02,-2.20706359e-01,1.90474167e-01,
1.97405607e-01,1.22916907e-01,1.35420829e-01,-3.40681598e-02,1.26317129e-01,
-1.22503296e-01,-2.27416724e-01,2.33933821e-01,1.37144187e-02,2.13554367e-01,
1.48455977e-01,-1.06835231e-01,-8.52363259e-02,2.88624376e-01,-2.17255190e-01,
-6.46422654e-02,2.26058755e-02,2.72003084e-01,4.40024212e-02,-2.13956147e-01,
1.67596005e-02,4.24991883e-02,-6.38152510e-02,-2.23453447e-01,7.18579143e-02,
1.23213582e-01,-2.88706005e-01,-2.67453250e-02,5.02249934e-02,5.03033578e-01,
-1.33767813e-01,-2.53970027e-02,2.59396099e-02,1.94092840e-01,-8.77662376e-02,
-2.89433926e-01,1.02393128e-01,-6.01474242e-03,-2.95566261e-01,6.78113624e-02,
-3.01585466e-01,2.53716320e-01,-1.59508139e-01,3.33253038e-03,7.67948925e-02,
1.16187982e-01,1.26859806e-02,2.54694652e-02,-6.47757873e-02,-1.97991818e-01,
9.34621990e-02,8.37303475e-02,3.72669250e-02,-1.45372406e-01,-2.23369207e-02,
-5.84084019e-02,-3.86007354e-02,-1.39164209e-01,3.81136984e-02,-1.28317162e-01,
1.41748026e-01,-5.23319505e-02,8.94617513e-02,-4.71683852e-02,-4.32746857e-02,
2.01122230e-03,2.70898733e-02,1.33664813e-02,-1.22926608e-01,4.04849984e-02,
1.59102812e-01,-1.35362342e-01,-4.45137136e-02,-1.67519394e-02,6.43791556e-02,
2.42189481e-03,-5.65901957e-03,-1.54438391e-01,-1.13184616e-01,-7.12467134e-02,
-1.05574420e-02,7.01715350e-02,9.26488265e-02,1.15446150e-02,6.23607002e-02,
9.28365812e-02,5.67134134e-02,1.10121295e-01,9.09665413e-03,-8.64062458e-03,
-5.45861050e-02,-8.51502493e-02,2.50870381e-02,1.29514799e-01,-7.68448710e-02,
1.78401962e-01,4.47993465e-02,1.41933989e-02,5.30817807e-02,-1.07439309e-02,
-3.73059995e-02,-7.53779113e-02,-7.70715401e-02,1.07705873e-02,-5.32630607e-02,
1.44499063e-01,-2.10117519e-01,-1.17035039e-01,2.23868325e-01,-3.66124511e-02,
6.55474886e-02,2.00914264e-01,-3.51382494e-02,7.76872560e-02,2.09207442e-02,
1.37992837e-02,3.83137375e-01,1.68691248e-01,-4.45993692e-01,-7.50603303e-02,
3.76232386e-01,-1.20467417e-01,1.28271326e-01,1.43833950e-01,6.26825571e-01,
3.14314989e-03,-1.25371218e-01,-9.28760692e-02,3.18303883e-01,-5.25355972e-02,
-1.03250511e-01,-4.70699891e-02,2.75428910e-02,8.63419026e-02,2.85096802e-02,
-1.32381532e-03,1.70742244e-01,-1.64171487e-01,-9.00983885e-02,-1.77580357e-01,
1.09622953e-02,-4.57388873e-04,-4.17901307e-01,3.34120005e-01,1.65306404e-01,
3.02364840e-03,2.90780902e-01,2.12643281e-01,2.91306645e-01,6.47506714e-02,
-6.26961350e-01,-3.28675687e-01,1.43919200e-01,7.08476454e-02,-6.70999587e-02,
-2.04034261e-02,-2.00354651e-01,5.82381845e-01,5.09148896e-01,-1.02611937e-01,
7.66831934e-02,4.51697141e-01,-5.07930964e-02,-6.95270598e-02,4.35388386e-01,
2.16242015e-01,1.45108417e-01,1.79439187e-01,2.06698775e-02,8.32997262e-02,
-1.50333628e-01,-5.53971678e-02,5.03310204e-01,5.04868805e-01,6.81259707e-02,
-4.09688354e-02,-8.48142244e-03,1.74091458e-01,-9.83529687e-02,1.16198352e-02,
-1.95845496e-03,8.41930732e-02,-1.17240109e-01,1.09074906e-01,1.64295703e-01,
-1.56282946e-01,-4.73798439e-02,6.54575527e-02,3.79863232e-02,1.22311085e-01,
-5.35991043e-03,1.05227074e-02,1.81082748e-02,-5.66454642e-02,4.78470549e-02,
-5.30441254e-02,-1.30310252e-01,-9.63990092e-02,-5.33023998e-02,-3.48120145e-02,
1.08322129e-01,-2.27920674e-02,-1.71505570e-01,-1.91690996e-01,-4.15621698e-02,
-2.44240358e-01,3.59206088e-02,1.12452038e-01,-2.58211851e-01,1.85342208e-01,
1.04145996e-01,-1.18981428e-01,1.34742245e-01,-6.38117045e-02,-4.08325419e-02,
2.08405897e-01,1.66463763e-01,-1.95251852e-01,-3.50825310e-01,4.82812554e-01,
8.23756158e-02,9.61875692e-02,-8.25647786e-02,5.44313490e-01,1.92414969e-01,
4.76038232e-02,2.59847604e-02,2.01371610e-01,-6.07010163e-02,-1.21548288e-01,
4.53266919e-01,1.99591979e-01,2.67670602e-01,6.31598681e-02,-1.29224554e-01,
3.65303010e-01,2.60022120e-03,5.61544187e-02,1.57107979e-01,2.97132641e-01,
1.84853479e-01,-5.10563850e-02,6.35968102e-03,4.05141413e-02,-2.23443042e-02,
1.22416295e-01,-1.17324494e-01,2.77488865e-02,1.71573997e-01,-1.27148461e-02,
-1.62191615e-01,-1.14991888e-02,-1.21774580e-02,9.93501320e-02,1.73453063e-01,
-7.19994456e-02,3.08822036e-01,1.86650544e-01,-9.32826325e-02,-4.55065519e-02,
2.76529759e-01,-1.25951365e-01,-1.04979850e-01,-2.07047649e-02,1.02273047e-01,
-3.75662073e-02,-5.48206381e-02,2.53073305e-01,2.41719410e-01,-2.59760302e-02,
-5.31258695e-02,8.58712867e-02,1.74022108e-01,-4.43855703e-01,-2.52574772e-01,
-3.46817106e-01,3.22581381e-01,-5.87253273e-02,3.10262680e-01,-7.00176731e-02,
2.49898672e-01,5.36951497e-02,2.26498976e-01,2.55874664e-01,3.78330499e-01,
-2.14248687e-01,4.27087963e-01,-6.29407763e-02,9.90155805e-03,7.09848762e-01,
4.72877562e-01,-3.38037372e-01,-6.35399390e-03,2.44711921e-01,-2.31042176e-01,
9.42700580e-02,2.01148853e-01,3.91607136e-01,5.71975887e-01,4.07570042e-02,
1.16812006e-01,8.80054832e-01,-1.25466660e-01,7.94631336e-03,-7.78116360e-02,
7.21280813e-01,-1.16490953e-01,6.37334362e-02,-4.45464522e-01,-5.95883764e-02,
-8.71701837e-02,-4.08909693e-02,7.22684562e-02,1.87313277e-02,3.36629689e-01,
-3.77922893e-01,1.80494890e-01,-2.57188141e-01,-8.74235183e-02,-2.26244614e-01,
4.62484419e-01,-1.17010791e-02,2.37663239e-01,5.38478732e-01,-6.57972395e-02,
9.98539478e-02,4.69252877e-02,-1.25974178e-01,-8.89210850e-02,-3.12403888e-01,
9.46370959e-01,2.47211814e-01,-4.96691853e-01,-2.07554877e-01,-1.90004800e-02,
2.01775450e-02,-2.68799867e-02,-7.49883056e-02,-3.00974667e-01,-5.34163835e-03,
1.91321224e-01,-2.09048137e-01,-7.42604807e-02,-3.69787812e-01,3.93468514e-02,
-8.13435167e-02,-6.95055276e-02,-6.10350035e-02,-4.02023830e-03,-1.03739671e-01,
-8.62294436e-02,-8.44320282e-02,-8.67951140e-02,-2.14656785e-01,2.07156893e-02,
2.78967880e-02,2.70370692e-01,8.09344053e-02,2.41692993e-03,-1.52448071e-02,
-2.85499245e-02,5.75808957e-02,3.75797033e-01,2.32988909e-01,-5.50398715e-02,
2.54301190e-01,4.60138023e-02,5.57745658e-02,2.04337373e-01,-4.58012931e-02,
1.14779659e-02,1.97012439e-01,2.03741491e-01,2.33628064e-01,6.26205578e-02,
2.04578429e-01,-1.50076941e-01,1.12021081e-01,8.36772099e-02,-2.50799060e-01,
1.96752194e-02,-1.31802306e-01,-1.46215186e-01,-6.98481202e-02,2.11149305e-01,
-2.67564952e-02,1.38072491e-01,-1.17768999e-02,6.26265883e-01,1.55402496e-01,
4.66674380e-02,7.37929121e-02,5.46053171e-01,-5.34125008e-02,-1.96002394e-01,
2.96646625e-01,1.81014448e-01,-5.21793254e-02,2.29787976e-01,-4.74045007e-03,
3.57148126e-02,-1.84048340e-01,-6.76240176e-02,-6.23853244e-02,1.32524475e-01,
-6.55151665e-01,-1.24612466e-01,-8.06372315e-02,1.94632232e-01,4.40174490e-02,
-4.99088764e-02,-5.51824318e-03,-1.03163861e-01,7.97219649e-02,-5.53231984e-02,
-1.70327470e-01,-2.63510317e-01,-1.27209932e-01,-3.75873625e-01,-1.11331716e-01,
9.71858352e-02,-2.80499279e-01,1.30259231e-01,1.50075816e-02,6.76684007e-02,
-2.25643411e-01,-1.36420146e-01,1.00799818e-02,-2.16366604e-01,-2.47668698e-01,
-7.93758035e-02,4.89825308e-01,5.27987145e-02,-1.54958069e-01,4.20968771e-01,
-8.89020134e-03,-2.26994324e-02,-1.20087661e-01,-2.36973260e-02,7.35488161e-02,
1.06278453e-02,8.39004442e-02,1.10189788e-01,1.42720323e-02,3.46795060e-02,
-3.45350796e-04,9.58250090e-02,6.69697598e-02,8.49035606e-02,4.74781394e-02,
2.26746798e-02,-1.69686198e-01,-3.19701284e-02,1.57108828e-01,-1.35078639e-01,
3.00409738e-02,-1.85919385e-02,1.08859621e-01,-2.86641791e-02,-7.28720874e-02,
6.22877777e-02,-8.47957209e-02,-3.97228077e-03,9.97813419e-02,3.98537181e-02,
5.29468097e-02,-2.69044429e-01,6.00232556e-02,1.18613370e-01,-1.89020663e-01,
-4.70116436e-02,-2.36835890e-02,-3.39529842e-01,5.96591495e-02,-3.62307504e-02,
8.23980495e-02,1.96053982e-01,-4.56454977e-02,-1.09899320e-01,3.02413702e-01,
-3.29706073e-01,-2.15882599e-01,6.27510995e-02,-4.28373329e-02,-1.24310791e-01,
-3.79658304e-02,-1.17652692e-01,3.36393148e-01,4.03450757e-01,-4.72690314e-02,
3.46939042e-02,4.73755628e-01,-2.23219693e-01,6.94408566e-02,3.02727550e-01,
5.03212869e-01,2.11335540e-01,1.32514134e-01,-2.58173496e-01,-7.91938677e-02,
-2.73787584e-02,-9.66122970e-02,-1.17777042e-01,2.69057333e-01,2.59179682e-01,
-3.60504538e-01,1.58802748e-01,5.56499138e-03,1.31200701e-01,1.88736066e-01,
-1.38644904e-01,4.74713892e-02,1.43566072e-01,-2.27498591e-01,-1.86900467e-01,
4.26037274e-02,1.19212851e-01,2.05740277e-02,-1.42568037e-01,-1.02039628e-01,
4.19704288e-01,1.67057022e-01,1.00925580e-01,3.81564125e-02,4.87802505e-01,
8.45584646e-02,-1.31082654e-01,3.04878086e-01,4.88538563e-01,-1.59893021e-01,
-8.60780105e-02,8.93810019e-03,3.27605635e-01,-1.96714491e-01,1.35212809e-01,
1.11282587e-01,2.03498542e-01,1.55252442e-02,-1.56274185e-01,1.15108170e-01,
3.04747880e-01,2.67090797e-01,1.61025241e-01,-1.69211805e-01,1.15044624e-01,
-1.61889821e-01,3.64359282e-02,6.00423329e-02,8.08547214e-02,1.78794339e-01,
1.06139831e-01,2.15886235e-01,-6.90365210e-02,6.22293167e-02,-1.46493599e-01,
8.40673107e-04,1.42757118e-01,-8.85130838e-02,3.30209881e-02,-1.87344961e-02,
2.78039634e-01,-1.90125674e-01,-1.29096672e-01,-4.00190860e-01,1.53321072e-01,
-3.82753238e-02,-1.05308637e-01,6.30418137e-02,-1.85069535e-02,-2.35754415e-01,
2.88069807e-02,-3.06349695e-01,-2.50062823e-01,-2.95113418e-02,2.74251908e-01,
-3.27588851e-03,3.76812667e-02,-1.75537616e-01,2.35137343e-01,2.35006496e-01,
-1.10404842e-01,1.85534626e-01,-2.80628085e-01,2.62996674e-01,-1.90158680e-01,
2.44480297e-02,-3.99710029e-01,-1.46180108e-01,-1.41348504e-02,1.28775658e-02,
1.78519234e-01,2.29688480e-01,1.08592771e-01,-1.07040279e-01,-3.39317322e-01,
-1.94336101e-01,-2.57869422e-01,5.22148423e-02,-2.07532927e-01,-1.15381986e-01,
1.33355558e-01,-3.50185856e-02,-3.97116877e-02,-1.98833589e-02,3.72377157e-01,
-2.45372266e-01,4.01590429e-02,4.26648520e-02,3.51307951e-02,1.84023380e-02,
-5.61472699e-02,-1.76966786e-01,3.75770149e-03,9.06301364e-02,-2.93635637e-01,
-9.74997878e-04,3.95034179e-02,-7.38573372e-02,1.21155223e-02,2.33410358e-01,
-3.54951084e-01,7.78038651e-02,-6.22490272e-02,-3.97740215e-01,-2.88043469e-02,
-4.61282320e-02,-1.27052575e-01,-1.40912607e-01,-4.66694683e-01,-3.24710347e-02,
7.99874067e-02,-1.23498723e-01,6.33652136e-02,-1.79753639e-02,-1.84655085e-01,
2.69031730e-02,7.65772210e-03,3.62960637e-01,-8.59910622e-02,5.81954755e-02,
1.20631129e-01,-9.34016705e-02,-2.62710094e-01,-6.16329759e-02,1.60679191e-01,
1.99589312e-01,1.00063600e-01,1.26974732e-01,2.14723758e-02,-1.12524278e-01,
-1.94893740e-02,6.84658810e-02,-9.90587249e-02,-1.44014448e-01,5.68170398e-02,
5.57575263e-02,-8.30534920e-02,1.83846653e-01,2.16462053e-02,2.38425508e-02,
2.05385581e-01,3.79644223e-02,4.76785563e-03,1.30325519e-02,2.32801680e-02,
2.55939495e-02,-1.77763812e-02,1.45636918e-02,-1.11319445e-01,7.51146913e-01,
2.81879455e-01,-3.15174520e-01,-1.69008330e-01,-2.91869462e-01,-1.87465593e-01,
4.99406420e-02,1.88245162e-01,5.25462627e-01,-4.69396591e-01,9.25366059e-02,
-1.66202679e-01,-8.73235643e-01,-8.22269097e-02,-1.07181676e-01,-8.74802843e-03,
8.05666983e-01,-2.50860810e-01,1.09703466e-01,-1.33954734e-01,-4.51480806e-01,
-9.40001756e-02,1.10582791e-01,1.53623685e-01,3.11698973e-01,-3.27243030e-01,
2.05978796e-01,-2.55523697e-02,8.77902210e-02,1.66059986e-01,1.06616318e-01,
-2.22856015e-01,-5.31946048e-02,-7.13594407e-02,1.00409254e-01,5.83660230e-02,
-1.81134157e-02,8.30923170e-02,-6.95784315e-02,-1.15213081e-01,1.05467722e-01,
-8.90496559e-03,-8.68576914e-02,-7.13784918e-02,-3.34498957e-02,1.03520870e-01,
-1.11625734e-04,3.56586099e-01,-3.97055820e-02,-1.36964768e-01,-3.97686720e-01,
6.52133673e-02,1.59479275e-01,-5.10446489e-01,-1.98353902e-01,-3.81056368e-02,
-8.94837528e-02,-4.03944105e-01,-1.36525586e-01,-2.51432598e-01,-1.09783247e-01,
-3.36116970e-01,-1.30724251e-01,-1.28783837e-01,-2.00489119e-01,-1.51348114e-01,
-7.52677545e-02,1.65018737e-02,-6.07576221e-02,-2.46351101e-02,1.13322712e-01,
2.80003399e-02,2.24015093e-03,-5.55281006e-02,2.42040567e-02,-3.27405520e-02,
-8.57313424e-02,1.10380240e-02,-1.28355473e-01,-1.21462028e-02,-3.44203487e-02,
4.21812832e-02,1.45697277e-02,7.84842595e-02,-4.23219763e-02,-1.26437917e-01,
-1.72339957e-02,-7.90704638e-02,-2.63562873e-02,-3.25240865e-02,-7.66893923e-02,
-1.65008545e-01,-1.37656741e-02,4.29779999e-02,8.94322917e-02,-8.63715336e-02,
4.80419844e-02,-1.16550885e-01,4.61078435e-02,-3.49364698e-01,5.20569503e-01,
1.43467322e-01,-1.09239489e-01,-4.69701029e-02,4.22292985e-02,-6.08869158e-02,
-5.14196530e-02,1.49986297e-02,7.28251934e-02,6.77012950e-02,-1.50925204e-01,
-2.92011499e-01,2.06211442e-03,3.29114914e-01,-5.42576499e-02,2.75197923e-01,
3.79277160e-03,-1.39333472e-01,-4.15955968e-02,-3.23585659e-01,-6.72335774e-02,
-4.25409451e-02,-9.46594849e-02,-2.69080788e-01,7.00001866e-02,-1.52479127e-01,
-2.76898802e-03,-2.22951919e-01,1.13792136e-01,-4.68181968e-02,-4.10422951e-01,
-3.55922490e-01,1.70694306e-01,-2.65741572e-02,-1.01690292e-01,-2.10263059e-01,
1.88398436e-01,-2.42794394e-01,-8.02789330e-02,-3.42243165e-02,-1.61833882e-01,
-1.13204017e-01,-2.24124305e-02,2.48588204e-01,-1.38432518e-01,-3.14104110e-01,
-9.64774564e-02,-5.51539995e-02,-5.39681762e-02,2.65753776e-01,9.38994437e-02,
-5.06864488e-02,-3.95912379e-02,-1.10798515e-01,4.31032069e-02,-1.12359412e-01,
3.85379046e-01,8.61102249e-03,-2.41463676e-01,-1.40513301e-01,1.08408466e-01,
1.05770774e-01,-6.48872852e-02,-1.43508837e-02,-3.95463109e-02,4.48183231e-02,
4.42510724e-01,-2.57225037e-01,-1.32758409e-01,-1.11910976e-01,1.72399238e-01,
-5.10115549e-02,2.40474090e-01,-4.36767153e-02,1.78628489e-01,2.56564736e-01,
1.67773589e-01,-2.82189190e-01,-9.63371620e-02,-5.49077690e-02,2.04562154e-02,
-2.96349019e-01,-1.35055006e-01,-1.01586087e-02,-7.87863657e-02,-4.04823691e-01,
2.93110628e-02,3.87570113e-02,-5.59840202e-01,3.01786028e-02,-2.60989904e-01,
-1.58807710e-01,7.97761604e-02,6.79670193e-04,-2.63880659e-02,1.50392890e-01,
1.28056303e-01,-2.81697541e-01,7.41397366e-02,1.79502532e-01,-1.27959758e-01,
2.30008867e-02,-4.88366522e-02,-1.33400232e-01,5.05268425e-02,6.80303574e-02,
1.21139303e-01,2.00561330e-01,-7.66020343e-02,1.35520011e-01,-1.43841296e-01,
3.34946960e-02,5.94771095e-02,9.30336043e-02,-5.36838286e-02,-5.41271009e-02,
9.65754595e-03,-4.47754860e-02,-2.27238946e-02,-6.91628680e-02,-4.86342758e-02,
-7.07702413e-02,-5.44732213e-02,-2.23236650e-01,-1.60817608e-01,6.62903860e-02,
-8.68769437e-02,1.46804899e-01,-1.48778260e-01,5.80789447e-02,-2.52103478e-01,
-4.78968859e-01,-2.12900620e-02,2.29330897e-01,2.12578073e-01,1.81869835e-01,
-1.11642696e-01,-6.96801618e-02,1.81017816e-01,-5.17623723e-01,-6.55160099e-02,
4.93392497e-02,-6.72751740e-02,-5.45526206e-01,-8.96736756e-02,2.64669001e-01,
1.04847215e-01,-7.19684303e-01,-1.73513427e-01,-1.64032485e-02,4.68015522e-02,
-1.67355880e-01,6.89644441e-02,-1.08444504e-01,1.46142453e-01,-1.48092583e-01,
2.42843077e-01,-7.56483078e-02,-4.89123315e-02,2.67025586e-02,1.12125531e-01,
-7.43890628e-02,-5.75521253e-02,1.65453523e-01,-7.56245013e-03,2.75002688e-01,
-3.70387882e-01,-2.07241233e-02,2.92650461e-01,2.61909157e-01,-8.77004638e-02,
4.76200171e-02,-2.17539594e-01,8.39525312e-02,-1.56996138e-02,1.15338393e-01,
-1.19171016e-01,-9.60804820e-02,-1.52280271e-01,5.69182895e-02,1.70037284e-01,
6.42862096e-02,-1.05189130e-01,6.46215305e-02,-7.24765435e-02,2.39855200e-02,
-1.56747848e-02,-2.45822161e-01,2.58868355e-02,-7.52245784e-02,6.04429059e-02,
3.25618610e-02,-7.16611326e-01,5.95833883e-02,2.34234575e-02,-6.26731515e-02,
9.10549704e-03,-2.70780772e-01,5.79480687e-03,5.28348207e-01,5.52996695e-01,
-2.61621684e-01,7.09590390e-02,1.87437888e-02,1.64508447e-01,5.72762713e-02,
-6.47253171e-02,8.96037444e-02,-1.11977309e-01,-3.32838744e-01,3.31724018e-01,
1.05306372e-01,-3.13158035e-02,-8.00261796e-02,8.14504400e-02,-8.95798132e-02,
-4.34019528e-02,9.75707099e-02,3.55561167e-01,1.33081019e-01,2.78806388e-01,
1.47614107e-02,2.73428261e-01,8.00102875e-02,4.43320796e-02,5.85807145e-01,
3.19046706e-01,-1.58970878e-01,1.16729639e-01,3.28963161e-01,-1.41956955e-01,
1.58229157e-01,4.64908965e-02,-1.03012063e-01,-1.11618284e-02,-3.88796628e-02,
8.76653641e-02,3.77378566e-03,-5.28681697e-03,2.10887380e-02,9.46478732e-03,
-5.70134968e-02,-1.93591550e-01,-2.72989810e-01,-1.03161000e-01,-2.33026873e-02,
-1.33860722e-01,-1.40894338e-01,-7.57329389e-02,3.09789181e-01,-4.27069277e-01,
2.88357772e-02,8.95758532e-03,-4.29268450e-01,7.97244832e-02,-4.02980179e-01,
-9.93234292e-03,3.44308019e-02,-7.70610049e-02,-3.76367569e-02,6.27266243e-02,
-8.66155699e-02,-1.24152578e-01,6.66398555e-02,-4.79462417e-03,-5.46970442e-02,
-7.05045611e-02,3.94475907e-02,-1.99160888e-03,-2.13517882e-02,1.89771265e-01,
1.17803132e-02,-8.79571214e-02,3.40405442e-02,-1.66616037e-01,8.71421695e-02,
4.23340723e-02,6.30381051e-03,-7.48099312e-02,-2.73336619e-01,-1.20215714e-01,
-1.82895847e-02,-4.02360745e-02,-8.24104697e-02,5.23857847e-02,1.88118909e-02,
-1.65336400e-01,1.29288873e-02,2.02994198e-01,-1.80357303e-02,2.52875108e-02,
1.70264080e-01,-2.11825818e-01,-4.87651750e-02,-1.38556376e-01,-3.13190483e-02,
9.78185087e-02,4.04073447e-01,-2.36820206e-01,-1.58779785e-01,2.01138377e-01,
-1.03460573e-01,-4.09684581e-04,3.12627643e-01,2.40927055e-01,9.29651260e-02,
2.08253160e-01,2.93274134e-01,1.60726048e-02,-1.66382343e-01,-7.91038647e-02,
1.24712944e-01,-1.31333768e-01,-3.62533242e-01,1.26594782e-01,-1.32439807e-02,
-1.66480407e-01,-4.77771536e-02,9.61508378e-02,-1.40158862e-01,-8.60748589e-02,
-2.52461344e-01,5.99490166e-01,2.88953245e-01,-1.53802216e-01,-1.73036426e-01,
5.94206601e-02,-2.29267970e-01,-6.61363453e-02,1.14887461e-01,3.20200115e-01,
-3.09624523e-02,-2.01109182e-02,2.33989015e-01,-1.00406304e-01,-3.14639434e-02,
-1.18833527e-01,4.11367714e-01,1.28380179e-01,2.07193032e-01,-7.41376653e-02,
1.15806930e-01,-4.09146622e-02,-4.75211948e-01,-3.99401933e-02,8.89517218e-02,
5.49227372e-02,-1.18191689e-01,-2.86051054e-02,1.81238949e-01,-1.54977903e-01,
7.44814202e-02,-4.17813696e-02,-5.06659448e-02,1.67837381e-01,-4.33770597e-01,
1.46935761e-01,-4.07659709e-02,4.41614650e-02,-2.45173816e-02,1.47424117e-02,
-7.67614618e-02,-8.93490110e-03,2.23422702e-02,-5.97163253e-02,8.25136155e-02,
-2.29398087e-02,-4.57303599e-02,8.07836950e-02,8.09754208e-02,5.72051518e-02,
-1.04098739e-02,-1.67161748e-02,1.93277486e-02,-3.83332819e-02,-1.54928379e-02,
6.82993829e-02,-6.49050996e-02,7.02849925e-02,-7.78572401e-03,8.11249614e-02,
4.81505543e-02,-2.66914107e-02,-1.29149899e-01,3.94113660e-02,-1.26529280e-02,
-1.81202278e-01,1.46966865e-02,2.99843550e-02,-2.14590225e-02,1.04543671e-01,
-1.48740830e-02,1.37661189e-01,-4.08176482e-02,-4.80868816e-02,6.56024143e-02,
-6.21859618e-02,-5.29678948e-02,-9.08295140e-02,1.22108795e-02,-1.09080553e-01,
7.92440176e-02,-2.97244713e-02,-7.31895044e-02,-1.62554737e-02,-8.02774876e-02,
4.80692759e-02,-1.70270920e-01,-1.04157217e-01,-6.12316877e-02,3.32960635e-02,
3.73974629e-02,3.19286925e-03,4.30173613e-02,-2.76355371e-02,2.53599614e-01,
1.60121739e-01,5.59395440e-02,-1.46969676e-03,-2.51570523e-01,1.61909595e-01,
-7.69638270e-02,3.43993232e-02,4.77337688e-02,-4.32897024e-02,-5.40024787e-02,
8.31016228e-02,-5.79885468e-02,2.14740232e-01,-6.71292916e-02,-6.30904064e-02,
-5.22300676e-02,2.30236351e-01,-3.38671240e-03,1.11185342e-01,-6.21802348e-04,
1.53631434e-01,1.35661915e-01,3.33019681e-02,-1.26102969e-01,-6.86531216e-02,
-1.47425711e-01,-3.51678252e-01,6.91543445e-02,3.41104150e-01,-4.41542491e-02,
-4.19413336e-02,-2.45185457e-02,-6.53366894e-02,4.52907756e-02,-6.74692169e-02,
-2.29127944e-01,1.66007325e-01,-7.36631453e-02,3.92720133e-01,-2.98284758e-02,
-2.72728860e-01,-3.10228486e-02,-2.92606819e-02,9.48785543e-02,-1.81555167e-01,
-8.42153281e-02,2.40068525e-01,1.57223359e-01,-6.83387965e-02,3.03779170e-02,
-1.41873453e-02,-8.26452896e-02,7.31164068e-02,4.80668604e-01,1.64344497e-02,
-6.66984171e-02,-7.56733939e-02,4.63213064e-02,9.71592441e-02,1.18952775e-02,
6.25480711e-02,6.05790913e-01,2.22311988e-01,-1.66251436e-02,1.23724587e-01,
2.08442926e-01,-2.34525308e-01,9.60906669e-02,1.35449976e-01,3.14406492e-02,
1.22496195e-01,5.18819802e-02,-1.01844080e-01,-1.29265860e-01,-1.53375822e-05,
1.21734232e-01,6.39091581e-02,1.33465737e-01,1.57282725e-02,2.99359169e-02,
-1.27165824e-01,7.45483786e-02,4.31523360e-02,1.27149999e-01,3.67175192e-02,
-2.96204700e-03,-6.00337237e-02,-6.03969842e-02,-1.87730566e-01,2.70354122e-01,
8.57055709e-02,1.04265496e-01,-7.98170269e-02,-6.70215115e-02,1.94649417e-02,
1.40365725e-02,4.84632403e-02,4.17635543e-03,7.98251703e-02,7.18804747e-02,
-8.99550132e-03,-3.09439171e-02,2.62505021e-02,-3.72110051e-03,-1.14371516e-01,
-6.73530549e-02,3.06653306e-02,1.12449683e-01,-2.94891931e-02,3.79715674e-02,
-8.26159120e-03,2.12911349e-02,2.45071679e-01,-2.54759520e-01,-2.67750114e-01,
5.97409299e-03,-1.09645873e-01,-5.98055907e-02,9.24864337e-02,5.71822412e-02,
3.40326548e-01,8.62009600e-02,1.95902631e-01,-1.19419724e-01,3.95359211e-02,
-5.09517014e-01,-1.03638545e-01,1.93224959e-02,1.86513811e-01,-1.32952303e-01,
-6.84106257e-03,-2.02215329e-01,-2.37149760e-01,9.19698700e-02,-2.13136658e-01,
-6.46563321e-02,-2.37581238e-01,2.80745059e-01,4.82647493e-02,-3.28609049e-02,
3.26632559e-02,3.16383988e-02,-1.35401502e-01,-7.27322474e-02,1.63938738e-02,
2.34726578e-01,-8.20398703e-03,-1.80812687e-01,4.90470320e-01,-1.59612775e-01,
4.25942875e-02,1.20516278e-01,-3.09889857e-02,6.18067205e-01,3.98930386e-02,
-2.20854267e-01,-3.18028063e-01,-2.13008285e-01,-5.84919453e-02,-9.25893709e-03,
-7.70057067e-02,-1.53860062e-01,1.40241995e-01,1.12521976e-01,-1.97783053e-01,
-1.14836641e-01,-8.22220221e-02,-4.26468849e-01,-7.27767199e-02,-3.11361626e-02,
3.41997534e-01,1.79959655e-01,-4.54525948e-02,-1.28807217e-01,2.79290862e-02,
-8.00922513e-02,-1.74429178e-01,-4.19198610e-02,-2.83790613e-03,-2.90915281e-01,
2.49889512e-02,-1.09777927e-01,1.52524829e-01,-1.03247603e-02,-1.48969203e-01,
4.17066753e-01,-1.13020435e-01,1.29160225e-01,-1.47496620e-02,1.56461850e-01,
-2.50559449e-01,2.25872234e-01,-1.24499597e-01,5.97168863e-01,-1.19135238e-01,
-1.30923942e-01,-8.33189636e-02,1.50558844e-01,1.19742937e-02,-7.90980756e-02,
-2.53899246e-01,4.25901026e-01,1.65658683e-01,1.80340022e-01,2.47009825e-02,
3.08096558e-01,-1.06448665e-01,6.04574531e-02,-8.01975876e-02,1.61002651e-01,
1.39439136e-01,2.46039405e-01,-5.82938716e-02,1.14282355e-01,-2.12838054e-01,
4.40217294e-02,-7.49217570e-02,7.96615146e-04,5.59736907e-01,-1.13057934e-01,
1.05053470e-01,5.62581830e-02,1.67073961e-02,-2.70226061e-01,-3.17396134e-01,
-1.50556698e-01,8.50204170e-01,8.34234834e-01,8.90886262e-02,8.63650441e-02,
2.00844392e-01,1.33440807e-01,3.60791385e-02,-2.69501299e-01,5.55755258e-01,
4.29274470e-01,-2.58743197e-01,5.67194261e-02,2.97147870e-01,1.65571962e-02,
4.37850654e-02,2.50715494e-01,2.14680880e-01,-7.24895671e-02,9.63701773e-03,
-7.40548149e-02,-1.30600035e-01,-3.88364570e-04,3.77007313e-02,-1.71290681e-01,
-3.29622962e-02,-5.81392273e-02,3.68432365e-02,4.88901772e-02,4.71785516e-02,
-2.91832417e-01,3.83190215e-02,5.47444113e-02,-1.28977709e-02,1.13196298e-01,
1.38969854e-01,-1.52039081e-01,1.07816011e-01,-1.99093949e-02,-2.81252223e-03,
1.40932471e-01,3.20071787e-01,-3.94109301e-02,8.03267509e-02,-4.39102575e-03,
-1.57237068e-01,-1.48052320e-01,-4.68638130e-02,-6.75900578e-02,8.73852223e-02,
-1.72905058e-01,4.25712802e-02,2.95249104e-01,6.97604194e-02,1.38277665e-01,
1.44612312e-01,-1.36928201e-01,-2.25970864e-01,5.11420541e-04,-1.15419544e-01,
3.16876501e-01,9.09852237e-03,-7.13824853e-02,-4.32061814e-02,-2.85721332e-01,
1.34924412e-01,1.76626798e-02,-1.06215343e-01,-6.53783698e-03,-2.37595383e-02,
-8.56885612e-02,3.35030675e-01,-1.02575764e-01,-3.23674046e-02,5.47767058e-03,
3.18549424e-01,-2.56708950e-01,2.62699276e-01,1.39400959e-01,7.43036196e-02,
-1.36165563e-02,-3.90832722e-02,1.58216804e-01,-7.41596073e-02,-3.27569172e-02,
3.77725139e-02,-8.90105888e-02,-3.87063809e-02,-1.04378045e-01,-2.89471477e-01,
1.20788151e-02,1.01726733e-01,-7.47572556e-02,-1.20222934e-01,8.93856362e-02,
-8.04037526e-02,-3.50333363e-01,-3.21842164e-01,-1.39066979e-01,2.38305926e-01,
-7.01189339e-02,2.65660405e-01,5.38206361e-02,-6.16569668e-02,2.39809304e-02,
-1.26359865e-01,6.22874014e-02,-6.64341450e-02,3.04640681e-01,2.51944423e-01,
-5.10747358e-03,5.38787767e-02,-1.08861201e-01,1.67158142e-01,-2.28030980e-01,
5.17538525e-02,-7.14864489e-03,-1.71257839e-01,5.89797460e-02,3.37554328e-02,
-8.16103145e-02,-1.59722026e-02,-3.53987925e-02,-3.16460207e-02,-3.77867371e-02,
1.74236055e-02,-8.43322650e-02,3.40279229e-02,-1.74547068e-03,-8.64487514e-03,
-1.33283511e-01,1.16289780e-01,-5.18469512e-02,-1.12717763e-01,-3.82108271e-01,
1.62559450e-02,1.05989859e-01,1.21658787e-01,-1.64412651e-02,7.96101056e-03,
-1.90651044e-03,-4.48700339e-02,-8.29874128e-02,-2.84605747e-04,-3.98784205e-02,
-1.04792759e-01,-2.49733450e-03,-7.14714155e-02,-1.12284273e-01,3.45882028e-02,
1.76466748e-01,1.53002426e-01,-1.69912860e-01,-1.17036782e-01,1.70944735e-01,
-6.92457780e-02,-1.06924094e-01,1.37007833e-01,4.53039408e-01,-1.20438799e-01,
1.05748691e-01,-6.32290766e-02,-2.37078339e-01,8.43985379e-02,-2.04588428e-01,
8.63851160e-02,-2.29107798e-03,3.40762079e-01,-1.29539981e-01,-4.99870221e-04,
6.69455901e-02,-2.59693772e-01,-1.22768074e-01,-2.57438660e-01,3.46161455e-01,
-3.43752265e-01,2.32751817e-01,4.37904000e-02,-1.93264082e-01,1.81877762e-01,
5.92761673e-02,-2.06743881e-01,-2.45750904e-01,1.23178743e-01,-1.79216322e-02,
-1.32845901e-02,9.49502587e-02,2.25318119e-01,1.25588715e-01,-9.36861485e-02,
5.16385101e-02,1.04170322e-01,2.76945472e-01,-4.57975566e-02,1.76876399e-03,
4.72725965e-02,-9.81220379e-02,-1.93948776e-01,-4.61888611e-02,2.86670804e-01,
2.15897188e-02,1.00788094e-01,-4.73664068e-02,2.50131249e-01,-1.12712853e-01,
-6.76796436e-02,1.89759225e-01,8.55771899e-02,-8.49193111e-02,1.46234229e-01,
1.53172240e-01,1.69447791e-02,1.83755308e-02,8.00840855e-02,-2.51329280e-02,
-3.46063495e-01,}; 
k2c_tensor lstm_1_recurrent_kernel = {&lstm_1_recurrent_kernel_array[0],2,4356,{132, 33,  1,  1,  1}}; 
float lstm_1_bias_array[132] = {
1.35313228e-01,1.82184502e-01,-2.43857410e-02,-1.35583714e-01,1.23553880e-01,
2.91562304e-02,-2.38292608e-02,-7.52168596e-02,4.14142832e-02,1.82291448e-01,
2.75733292e-01,2.92031288e-01,3.95300388e-02,1.34776995e-01,9.77720320e-02,
-1.00697074e-02,2.29098901e-01,1.56947002e-01,5.58970757e-02,2.82632709e-02,
2.00469032e-01,2.04111546e-01,-4.31329906e-02,1.80082917e-01,2.27375448e-01,
2.11236387e-01,4.76142287e-01,1.74363345e-01,-3.42948250e-02,3.00493717e-01,
-5.58665255e-03,2.42473669e-02,-1.33749573e-02,9.39729095e-01,1.06013548e+00,
1.09859133e+00,9.03023958e-01,9.11366999e-01,1.27345538e+00,9.79464650e-01,
7.74414957e-01,9.85080838e-01,8.80788803e-01,1.07555830e+00,1.12243927e+00,
1.14577127e+00,1.21850789e+00,1.28544843e+00,9.91789997e-01,1.27539659e+00,
9.07791615e-01,1.08194816e+00,1.04383385e+00,1.17476344e+00,1.37694001e+00,
9.52971876e-01,1.48916018e+00,1.16258788e+00,1.23845649e+00,9.08796489e-01,
1.03307724e+00,1.19105816e+00,9.28096235e-01,9.90028441e-01,1.37404513e+00,
1.33428192e+00,-2.02747002e-01,1.23328790e-01,1.85693651e-01,1.25960633e-01,
1.27622902e-01,-7.70599321e-02,-8.26971680e-02,-1.75380915e-01,-1.47729561e-01,
-1.66099407e-02,-4.99927849e-02,4.78178076e-02,1.19548328e-01,-2.45045368e-02,
3.78507674e-02,-1.75647244e-01,1.51389137e-01,-1.14256874e-01,-1.57449096e-01,
-1.13366432e-01,3.61696750e-01,-1.66281670e-01,-1.96278080e-01,3.25453907e-01,
-2.10341364e-02,-2.42092863e-01,-1.35313854e-01,-1.79216057e-01,1.69607401e-01,
-1.59590676e-01,-1.20764598e-01,2.58027107e-01,2.36415744e-01,6.67481348e-02,
1.93395376e-01,-1.47495151e-01,-2.03976586e-01,8.63504484e-02,-7.32964724e-02,
-2.80651748e-02,1.02338813e-01,-1.80404693e-01,1.75621718e-01,3.53007793e-01,
2.59473413e-01,-9.81361046e-02,2.91311890e-01,-1.82465881e-01,-1.24910418e-02,
1.61324635e-01,9.85060632e-02,3.21103334e-02,1.83831342e-02,1.20872006e-01,
1.46140292e-01,-3.71598676e-02,1.80748448e-01,2.71066338e-01,-1.40933692e-02,
3.05921018e-01,1.81995049e-01,-1.21476203e-01,3.27342808e-01,-5.97529486e-03,
6.42755479e-02,1.01041943e-01,}; 
k2c_tensor lstm_1_bias = {&lstm_1_bias_array[0],1,132,{132,  1,  1,  1,  1}}; 

 
float lstm_2_output_array[33] = {0}; 
k2c_tensor lstm_2_output = {&lstm_2_output_array[0],1,33,{33, 1, 1, 1, 1}}; 
float lstm_2_fwork[264] = {0}; 
int lstm_2_go_backwards = 0;
int lstm_2_return_sequences = 0;
float lstm_2_state[66] = {0}; 
float lstm_2_kernel_array[528] = {
-4.74272162e-01,-9.87793654e-02,-4.71979320e-01,9.16994736e-02,-7.13059485e-01,
-6.42329395e-01,1.98767766e-01,-9.67190489e-02,-3.32107633e-01,-4.76641655e-01,
-4.51146718e-03,-3.75762045e-01,-1.11493520e-01,-3.86848778e-01,-2.00007498e-01,
-2.14097366e-01,-4.39697981e-01,1.94432326e-02,-3.65944147e-01,-7.70419598e-01,
-5.55720568e-01,-3.61707471e-02,-5.17232656e-01,-2.66192317e-01,4.24585372e-01,
-7.99608469e-01,-3.81259412e-01,3.09320420e-01,-5.39445341e-01,-2.80211776e-01,
-2.81419545e-01,-3.72962683e-01,-1.00951564e+00,-5.51443361e-02,-2.40837604e-01,
-2.21589193e-01,-4.84581366e-02,-4.56289351e-01,-3.93934250e-01,-3.71862948e-02,
-1.92700475e-02,-1.02333689e+00,-1.50147989e-01,-2.94361770e-01,-4.96058941e-01,
5.54771066e-01,3.71875912e-01,-1.25619650e-01,1.76371392e-02,-4.60879087e-01,
-3.19956571e-01,-2.87071198e-01,-3.63431722e-01,-3.33469003e-01,-2.64179766e-01,
4.81601447e-01,3.10643036e-02,8.50572586e-02,-4.33206648e-01,-4.97329086e-01,
-1.02978516e+00,1.55977860e-01,-4.30920422e-01,-3.27765256e-01,5.16354620e-01,
-3.21456313e-01,-2.69411802e-01,7.67281055e-02,-1.01548769e-01,5.63495569e-02,
-4.32235628e-01,-1.23980209e-01,2.03011617e-01,5.64751439e-02,-2.30124131e-01,
2.63972040e-02,-1.41928136e-01,3.73714417e-02,-7.18946010e-02,-3.54370810e-02,
-6.90984949e-02,-3.20126027e-01,-2.62643456e-01,2.48119727e-01,-1.84395984e-01,
-2.59917289e-01,-2.96467960e-01,-4.27331477e-02,3.00089419e-02,-3.88368398e-01,
1.45948336e-01,-2.00104311e-01,-7.65075743e-01,-1.51970193e-01,5.43978512e-01,
-1.38218077e-02,-2.47338042e-01,-2.45805487e-01,-5.28226560e-03,-2.57208478e-02,
1.38296470e-01,-1.83924466e-01,-8.24181139e-02,9.53170434e-02,-2.41737261e-01,
-1.87461488e-02,-2.23130465e-01,-3.21304239e-02,2.06545755e-01,-1.32621154e-02,
1.00380577e-01,-3.03641200e-01,4.14174706e-01,-1.98137179e-01,-5.25592923e-01,
2.09375426e-01,-7.85522014e-02,1.59871861e-01,6.94043458e-01,-3.71449471e-01,
3.07717383e-01,4.61941421e-01,1.50460809e-01,7.12585688e-01,-6.75429106e-02,
2.13111207e-01,-1.36809284e-02,6.85248012e-03,-2.19055101e-01,2.18842864e-01,
4.68790799e-01,-1.15953669e-01,2.93244064e-01,1.49166966e-02,-2.06650943e-01,
-5.23597002e-02,-1.42207472e-02,-5.73060140e-02,2.71937698e-01,3.83083187e-02,
3.68827969e-01,-1.94868110e-02,4.86937493e-01,-3.62142511e-02,4.32465643e-01,
-1.55859366e-01,3.43212456e-01,3.48694414e-01,5.05167902e-01,1.13120303e-01,
3.77954274e-01,5.75210810e-01,2.85668194e-01,3.33340526e-01,1.60651147e-01,
8.32442939e-02,2.15644866e-01,5.63189447e-01,-1.89859957e-01,2.80262828e-02,
1.45812221e-02,-2.00684860e-01,1.42504066e-01,-2.01586604e-01,-6.20089397e-02,
2.03654483e-01,2.17025355e-01,1.62530437e-01,1.13145836e-01,3.36626679e-01,
4.71674591e-01,-3.37064892e-01,3.31339389e-01,1.00624931e+00,4.18836251e-02,
7.41631314e-02,-1.70050815e-01,7.36462295e-01,-5.07147670e-01,-8.59780759e-02,
-5.80066517e-02,6.46297812e-01,1.42558739e-01,7.81526864e-02,-1.63019657e-01,
-2.71079957e-01,-3.31640422e-01,1.37600238e-02,7.12414861e-01,4.08822417e-01,
-2.50827938e-01,2.18673572e-01,4.79120553e-01,3.65962863e-01,-1.14187323e-01,
-2.88555026e-01,8.03978026e-01,1.11861143e-03,2.72634476e-01,6.42594099e-02,
2.57913589e-01,-1.58273131e-01,-5.46640940e-02,-2.71223247e-01,1.48545533e-01,
3.66574794e-01,2.12333202e-02,-5.65957204e-02,2.18663946e-01,-4.53079671e-01,
2.75558740e-01,-9.28737968e-02,6.40453249e-02,3.52795511e-01,3.10798913e-01,
-1.21798187e-01,4.57758218e-01,4.11393285e-01,4.42230344e-01,1.00506335e-01,
6.24306440e-01,3.83765437e-02,-2.25093722e-01,1.40964007e+00,2.62642503e-01,
4.36122082e-02,3.05543214e-01,3.26951556e-02,1.66220088e-02,-5.25294304e-01,
2.88551629e-01,-1.10700585e-01,-1.21838413e-02,-1.48412157e-02,5.63020110e-02,
1.64584741e-02,-2.80945897e-01,-1.39252767e-01,-3.20243388e-01,-5.44242524e-02,
1.53196761e-02,-1.23824306e-01,-1.12782940e-01,-2.82503851e-02,-4.78096167e-03,
1.39619187e-01,-9.15416002e-01,-5.33727288e-01,-1.07204594e-01,-9.80745926e-02,
-2.11447924e-01,5.48480265e-02,3.90331373e-02,-6.54755812e-03,4.25671697e-01,
1.24562613e-03,-6.21516109e-01,-9.51541245e-01,-2.35728592e-01,-2.81853348e-01,
-2.99259037e-01,-1.08225398e-01,1.57505542e-01,3.16495299e-01,-2.39617109e-01,
-1.89823993e-02,2.95609385e-01,-5.96747883e-02,4.15970773e-01,9.80348215e-02,
2.16364697e-01,8.29553679e-02,8.16140398e-02,3.46668251e-02,6.56377450e-02,
2.49324068e-01,-6.17667250e-02,-2.16568008e-01,5.45898438e-01,-8.05839431e-03,
2.22251132e-01,1.76223770e-01,3.13421756e-01,-4.98362221e-02,3.39362025e-01,
9.14506763e-02,4.05355915e-02,-1.11048467e-01,1.64938241e-01,3.13207239e-01,
5.02817929e-01,-3.83141674e-02,1.37090161e-01,3.81709844e-01,4.03629631e-01,
3.85438800e-01,6.13369226e-01,-9.20677483e-02,-1.62000880e-01,8.36253688e-02,
5.91245154e-03,9.94758010e-02,-8.74782875e-02,4.65907395e-01,-2.24816874e-02,
1.99505895e-01,-2.37807259e-02,5.07462919e-01,8.04107822e-03,1.68934882e-01,
3.57407033e-01,6.77004531e-02,3.33569705e-01,1.46362409e-01,-1.40317127e-01,
-3.10927164e-02,5.55257559e-01,-7.49732628e-02,6.09773338e-01,2.86318094e-01,
1.52033821e-01,1.17730498e-01,1.66205093e-01,6.14673132e-03,1.89457610e-01,
1.05608177e+00,-6.97725117e-02,-1.04604356e-01,3.94504040e-01,3.45348604e-02,
-2.72271723e-01,-1.81114644e-01,1.06022336e-01,1.03386350e-01,9.70974788e-02,
-1.27576679e-01,2.68272460e-01,4.69149910e-02,4.81022224e-02,-1.57230943e-02,
6.83815265e-03,2.00040817e-01,4.47134078e-02,1.43445164e-01,8.38265195e-02,
-4.50934982e-03,-1.21028267e-01,-4.22501527e-02,5.46436012e-02,2.75736451e-01,
3.33278929e-03,2.35229190e-02,-2.85070360e-01,4.52235192e-01,-1.37965316e-02,
-2.45735332e-01,1.09515436e-01,9.23327953e-02,-1.06004804e-01,-1.96227834e-01,
4.17665482e-01,1.44201696e-01,-2.25866169e-01,1.62357852e-01,2.25731265e-03,
2.73842532e-02,-1.81662589e-01,-8.20543543e-02,-5.85956536e-02,8.11655670e-02,
-1.64906919e-01,5.16984053e-02,2.03281596e-01,-3.07967179e-02,-3.00396960e-02,
1.38979092e-01,-1.42072722e-01,2.80285850e-02,-1.27757698e-01,-1.00714587e-01,
-1.30729806e+00,-2.93213367e-01,-1.39136270e-01,7.14494884e-02,5.39503396e-02,
-1.06765144e-01,-6.95834994e-01,4.03437942e-01,-6.98653221e-01,-1.87258542e-01,
3.40967998e-02,-1.98412210e-01,-5.12926519e-01,-9.92116891e-03,-6.00427724e-02,
-4.43199754e-01,1.47038147e-01,-3.17249894e-01,-3.89993966e-01,1.78404674e-01,
-4.16886598e-01,-2.22279161e-01,1.35546774e-01,-2.42009118e-01,2.68978953e-01,
-2.65045792e-01,-3.04641813e-01,-4.04920816e-01,-1.09567344e-01,-4.09214854e-01,
-3.58902425e-01,-1.22332424e-01,2.00309902e-01,-3.77636105e-02,-5.58860004e-01,
-6.89460635e-01,-2.68740803e-01,6.21225387e-02,-6.22303486e-01,1.00217694e-02,
3.08697730e-01,-2.77648538e-01,-1.02637792e+00,6.22869670e-01,-4.33306903e-01,
-3.93644005e-01,-1.11558205e-02,-8.16335604e-02,-8.15319240e-01,-8.75848830e-02,
-4.01744962e-01,-1.66048869e-01,-1.97911769e-01,-7.17908666e-02,4.69062142e-02,
-9.84083265e-02,6.50258437e-02,-2.16755241e-01,5.03047891e-02,3.05473637e-02,
-2.61593074e-01,8.67599428e-01,1.15758255e-01,-3.42779653e-03,2.04700604e-01,
4.48769182e-01,-5.07281661e-01,-1.27814993e-01,5.30718148e-01,-4.78723049e-01,
-1.19138777e-01,6.79527998e-01,4.38381374e-01,4.54607494e-02,2.48925779e-02,
-2.36675497e-02,7.44230151e-01,8.50205064e-01,-3.05438280e-01,-3.77514064e-01,
1.63721108e+00,-2.25353912e-01,6.41441122e-02,-2.45810241e-01,-4.70044240e-02,
-1.74236134e-01,-1.32608145e-01,-3.81045908e-01,-2.93875873e-01,-3.60985585e-02,
-2.34386042e-01,-1.35477543e-01,-1.24234341e-01,-5.73540807e-01,2.13064596e-01,
-5.34505695e-02,-3.73403579e-01,-1.34601533e-01,5.59933856e-02,3.15094069e-02,
-3.78417492e-01,-4.28901523e-01,1.73108697e-01,-2.22188190e-01,-7.76121691e-02,
-3.04311842e-01,8.52171257e-02,1.47261277e-01,-1.90519646e-01,2.65617728e-01,
1.22507354e-02,-2.33117398e-02,2.48399656e-03,-2.16845840e-01,-1.28718913e-01,
4.59184535e-02,1.17112644e-01,-3.35623585e-02,1.28599539e-01,3.04576248e-01,
-7.26380721e-02,1.96306054e-02,-9.40933675e-02,-2.06197485e-01,-5.28048724e-02,
-1.62855700e-01,3.71554941e-01,-1.66364580e-01,2.82352358e-01,3.27689677e-01,
-7.93465614e-01,-1.12130679e-03,1.23335414e-01,5.88767556e-03,8.40491056e-01,
-4.48903628e-03,3.68240207e-01,1.86966375e-01,3.42345655e-01,1.18062712e-01,
-3.56352121e-01,1.20633896e-02,-2.18701363e-01,2.00398192e-01,-2.35011548e-01,
8.20866466e-01,3.38218451e-01,3.52466968e-03,}; 
k2c_tensor lstm_2_kernel = {&lstm_2_kernel_array[0],2,528,{16,33, 1, 1, 1}}; 
float lstm_2_recurrent_kernel_array[4356] = {
6.55777846e-03,2.41978124e-01,2.15037197e-01,5.95693700e-02,7.14455470e-02,
-1.14412814e-01,-1.29053831e-01,-9.05624125e-03,1.51197270e-01,2.02380255e-01,
2.73166209e-01,2.33281612e-01,1.43499687e-01,1.67058751e-01,-1.02561876e-01,
3.08865577e-01,2.07123649e-03,1.63871229e-01,3.59390751e-02,3.56420092e-02,
1.96770474e-01,-9.47894156e-02,5.45045435e-02,2.01895356e-01,6.11898638e-02,
3.83684427e-01,-7.97448084e-02,4.11907703e-01,4.73959632e-02,9.91371125e-02,
3.29488665e-01,2.67382324e-01,9.93243456e-02,-1.96173549e-01,8.63862038e-02,
1.07834496e-01,9.99144465e-02,1.40984476e-01,1.16807759e-01,1.64163545e-01,
1.87946960e-01,-1.00884922e-02,-1.94076858e-02,7.85328522e-02,6.09436743e-02,
-1.40622795e-01,-4.70475527e-03,-1.36426374e-01,8.64957422e-02,-5.98773733e-02,
-9.28183421e-02,-1.91257037e-02,4.85398918e-02,1.57374769e-01,-8.41549635e-02,
-1.66606009e-01,-7.11728707e-02,9.35670286e-02,5.65602370e-02,-1.29731297e-01,
5.75823747e-02,2.83780247e-02,4.36649285e-02,1.09560445e-01,3.39044370e-02,
-9.52659175e-02,-7.75326565e-02,9.54265997e-04,1.65894911e-01,-8.18474889e-02,
1.81651935e-01,4.63890471e-02,-1.73249513e-01,-3.27418536e-01,1.38752192e-04,
-1.97855741e-01,7.60164186e-02,5.55840833e-03,1.97015144e-02,-7.44155869e-02,
7.86445066e-02,9.51158255e-02,-8.00355300e-02,-6.41225949e-02,-1.98510755e-02,
1.77422047e-01,-5.16004041e-02,3.98933798e-01,-1.24174587e-01,-2.79684156e-01,
8.07559043e-02,1.47585943e-01,-1.75479263e-01,9.47196037e-02,3.11037034e-01,
4.37478460e-02,-9.80007350e-02,-2.23718863e-03,-6.67573065e-02,1.65546983e-01,
-9.62786027e-04,1.15215458e-01,1.28532842e-01,2.62929965e-02,3.19428667e-02,
-2.93086078e-02,3.37683074e-02,-4.67861034e-02,-4.70847599e-02,-2.32401509e-02,
7.95058981e-02,-1.06881373e-01,-4.21545357e-02,6.27562776e-02,2.00576916e-01,
2.30751205e-02,-1.13799386e-01,-3.76583263e-02,-3.57845090e-02,1.73924286e-02,
9.33004543e-02,4.31571379e-02,-5.77603988e-02,-4.29024883e-02,-1.01836592e-01,
2.13656463e-02,-1.17200222e-02,-5.64501360e-02,-1.28233612e-01,-1.54437184e-01,
-1.72928497e-01,-1.61007226e-01,1.43870175e-01,7.24170804e-02,-6.42012656e-02,
8.71391147e-02,-2.43114196e-02,-1.83213763e-02,-1.27035782e-01,-2.52777725e-01,
4.77833860e-02,-3.16210181e-01,-6.18944243e-02,-6.39957041e-02,-7.26277083e-02,
-9.07437429e-02,5.81723265e-02,8.62860158e-02,2.97763437e-01,-1.25953499e-02,
5.23349904e-02,-5.04624844e-02,1.80683419e-01,1.47460163e-01,6.91864416e-02,
-1.45231858e-01,-1.15825452e-01,-1.31512448e-01,-3.61967355e-01,5.58822751e-02,
6.20009303e-01,-3.73253785e-02,5.19703189e-03,-3.91999073e-02,6.65568709e-02,
7.46717975e-02,5.36667509e-03,2.03902081e-01,2.60634976e-03,5.76135479e-02,
1.66765243e-01,7.66889006e-03,1.26962155e-01,7.26762339e-02,-2.97059193e-02,
3.28732468e-02,-1.12608641e-01,5.31167015e-02,-2.65285641e-01,1.66221946e-01,
3.91013660e-02,1.67625040e-01,1.34640425e-01,4.55026589e-02,1.88946381e-01,
4.23643226e-03,-5.22383302e-02,-8.70732963e-03,-1.82843599e-02,1.65768951e-01,
1.63245738e-01,7.16220811e-02,2.15463370e-01,2.74901897e-01,2.65501104e-02,
8.40195417e-02,3.21584493e-01,1.56810179e-01,4.23252881e-02,-1.08886115e-01,
2.83840060e-01,4.04885337e-02,3.87490392e-01,-5.44694699e-02,-1.40849665e-01,
-2.43599713e-01,1.24041378e-01,3.55660729e-02,2.00541466e-02,6.74826503e-02,
1.82503715e-01,-8.36377665e-02,-3.44199240e-02,2.39463039e-02,-1.29335076e-01,
-5.66191934e-02,4.15250147e-03,1.59159914e-01,9.43545923e-02,1.21006258e-01,
6.33410811e-02,-9.14541259e-02,-1.14301890e-01,1.66705877e-01,-1.15064621e-01,
-2.41287122e-03,1.81354925e-01,-2.39464752e-02,-7.85953552e-03,-6.54611513e-02,
2.53590196e-01,-1.45785976e-02,1.39964018e-02,2.33234867e-01,-5.05961739e-02,
2.41768092e-01,-6.11898229e-02,-6.63933679e-02,9.55630094e-02,2.63476342e-01,
7.45796561e-02,2.77729511e-01,1.13025039e-01,4.17692572e-01,2.20552355e-01,
2.69635260e-01,2.80226111e-01,3.20286036e-01,-1.06523223e-02,5.73333919e-01,
6.12576783e-01,1.55701935e-01,6.17365718e-01,1.59777761e-01,1.81216071e-03,
8.27717558e-02,3.68156195e-01,5.57722189e-02,2.50799268e-01,5.74597061e-01,
5.73173948e-02,9.67350304e-02,3.69269699e-01,2.32823104e-01,1.35249689e-01,
3.61144766e-02,3.02645117e-01,-1.28609195e-01,2.33975932e-01,4.86155152e-02,
-2.18436077e-01,3.60702872e-01,3.67024094e-02,1.03607315e-04,-1.01069212e-01,
1.46846324e-01,7.63008744e-02,-2.02035025e-01,4.12375867e-01,-6.06305599e-02,
1.94737360e-01,-1.26951501e-01,-1.80420086e-01,6.25612736e-02,2.41231829e-01,
-4.47001234e-02,6.03166968e-02,-1.86702777e-02,-1.25865042e-01,6.75698090e-03,
-7.62475803e-02,1.80845529e-01,3.47229421e-01,-5.35322540e-02,6.98956177e-02,
2.78516918e-01,3.10217083e-01,1.47959873e-01,1.59537509e-01,1.12980023e-01,
-3.19062956e-02,1.21839598e-01,2.76856236e-02,-2.97477216e-01,5.74416444e-02,
1.36209354e-01,-1.86826929e-01,-1.51192442e-01,-2.23662108e-01,4.97719012e-02,
1.91508476e-02,-8.46531168e-02,3.01625252e-01,1.29125595e-01,3.55475843e-01,
1.12140678e-01,3.53572994e-01,3.50842886e-02,1.14115819e-01,3.68917286e-02,
2.25984722e-01,1.28794223e-01,1.54537857e-01,-7.76244253e-02,3.43976438e-01,
1.06279165e-01,-2.45111976e-02,1.65929362e-01,1.68915361e-01,1.83536366e-01,
-1.68155923e-01,-2.69666929e-02,-2.11496442e-03,7.33425468e-02,2.37625957e-01,
-1.02995187e-01,-4.56427298e-02,-1.94875091e-01,7.21722767e-02,-1.64860878e-02,
-2.39016622e-01,3.47641222e-02,-1.92218479e-02,-1.58308700e-01,-5.61220292e-03,
1.11309901e-01,-2.34122220e-02,-1.25649543e-02,-4.63654846e-02,5.26644522e-03,
-2.56503999e-01,7.53429160e-02,3.23822759e-02,-7.45502263e-02,-3.66361775e-02,
2.56637663e-01,1.05793830e-02,2.18054596e-02,4.62609045e-02,-1.49831772e-02,
1.22753792e-01,1.17487565e-01,-7.06312358e-02,2.42570370e-01,3.45972441e-02,
-1.03653766e-01,6.31649047e-03,-2.39711683e-02,-1.05595611e-01,-6.37169778e-02,
1.61476657e-01,3.01467031e-02,5.80378473e-02,2.69569624e-02,-1.67258337e-01,
-3.07310104e-01,-2.34114259e-01,-5.89897148e-02,2.41545588e-02,9.53442529e-02,
-2.04566643e-02,5.70857711e-02,2.33947128e-01,1.40239000e-01,2.74067909e-01,
-7.20828846e-02,1.10183798e-01,-4.46433909e-02,1.24004134e-03,-2.96657264e-01,
2.17432857e-01,5.31511307e-01,-3.18964273e-02,-8.81978497e-02,1.58552691e-01,
-2.33753681e-01,1.29831642e-01,-9.35520455e-02,1.06939577e-01,3.78602259e-02,
5.55334762e-02,-5.57735227e-02,1.32178068e-02,-5.28387018e-02,-3.36912349e-02,
8.79096836e-02,-1.33504868e-01,-1.77888684e-02,-1.57782249e-02,-5.85119650e-02,
1.89327374e-02,-3.15075070e-02,-3.75694931e-02,1.27590477e-01,-1.80302083e-01,
-1.20367870e-01,-1.09732829e-01,1.86717898e-01,-2.06192404e-01,-4.21772059e-03,
5.12621319e-03,1.20046906e-01,-1.12634160e-01,1.14971779e-01,2.19251618e-01,
2.30783354e-02,2.58070499e-01,-5.11220694e-02,8.41542985e-03,9.77754816e-02,
1.02177076e-01,9.07561854e-02,-9.83565487e-03,4.58289012e-02,-6.12001829e-02,
-1.70116678e-01,-2.20798343e-01,-1.38586521e-01,6.67642727e-02,-5.27727939e-02,
-1.93396240e-01,-2.54528195e-01,-4.41506281e-02,-2.67095208e-01,6.81453273e-02,
1.01941124e-01,-1.15275204e-01,-1.61849931e-01,3.40360310e-03,-7.33395964e-02,
4.48498912e-02,1.21863738e-01,-1.79251209e-01,-4.69923615e-02,2.02770252e-02,
-1.94190055e-01,-4.32296991e-02,-2.78888792e-01,2.08951309e-01,-1.22294769e-01,
-5.33428192e-02,2.53485842e-03,-1.95523605e-01,-5.18617518e-02,1.04932878e-02,
-3.01811472e-02,6.65448755e-02,-2.57198602e-01,1.00869745e-01,-2.39130393e-01,
-8.63129944e-02,-1.10074908e-01,1.58611819e-01,-5.03962785e-02,2.81872116e-02,
-7.38039911e-02,5.62378801e-02,1.03381895e-01,9.12408829e-02,-1.49164535e-02,
-1.34012833e-01,-9.79213640e-02,-5.99043034e-02,2.25204274e-01,-2.63545275e-01,
-8.49202946e-02,-1.83911830e-01,-2.34236911e-01,-9.95318498e-03,-1.04803041e-01,
4.12466496e-01,1.69437706e-01,-2.08298326e-01,2.30591238e-01,8.19029287e-02,
-1.07368365e-01,-1.88165735e-02,2.18667760e-01,4.25132401e-02,1.26975611e-01,
-6.15514107e-02,5.01435548e-02,-1.88858643e-01,3.60019654e-02,-1.59895658e-01,
7.95258284e-02,-4.38202992e-02,7.56776035e-02,-1.59420504e-03,-1.40212938e-01,
2.23230377e-01,3.52583453e-02,2.06346456e-02,-3.53054702e-01,1.19362846e-02,
-3.99079829e-01,-3.92779894e-02,2.01470792e-01,-6.69427142e-02,-2.60951877e-01,
2.65064776e-01,-2.86971033e-01,4.45931196e-01,-9.26947594e-02,6.53164610e-02,
3.85796949e-02,7.55176395e-02,-3.25394534e-02,-1.61785632e-02,-2.21478447e-01,
1.28273085e-01,9.47234705e-02,1.48022354e-01,1.20240822e-01,-1.05641849e-01,
4.10248190e-02,1.02933133e-02,-1.93822533e-01,-5.31292260e-02,1.70285568e-01,
-4.59673293e-02,-1.72145590e-01,2.72645563e-01,1.23097531e-01,5.39287962e-02,
6.66541164e-04,7.18160346e-02,1.35874495e-01,2.09642854e-02,2.30005354e-01,
-6.20857775e-02,-1.73295841e-01,-1.01070404e-01,1.95038751e-01,-1.76573902e-01,
1.49794936e-01,3.58906955e-01,1.68142885e-01,-9.02855098e-02,1.68378904e-01,
1.96043700e-01,-3.65335941e-01,6.59158046e-04,5.32462038e-02,-2.83367217e-01,
1.27165735e-01,-8.42100848e-03,-6.71021566e-02,1.34880871e-01,1.61994412e-03,
2.34908029e-01,4.26256564e-03,-1.03567123e-01,-1.47223368e-01,-1.97977483e-01,
-2.19861060e-01,3.76821831e-02,-8.19229037e-02,-1.73158467e-01,-1.89712837e-01,
-2.10778102e-01,-2.02100262e-01,-8.46507587e-03,4.24063168e-02,4.25263047e-02,
2.29825333e-01,-1.24724612e-01,9.94360298e-02,-1.60984516e-01,2.03053225e-02,
1.01710387e-01,2.48013869e-01,-3.59558016e-02,-1.16664022e-02,-9.90086943e-02,
-4.71302308e-02,2.03235839e-02,1.74819678e-01,-7.66760856e-02,-1.48224592e-01,
-1.34473696e-01,-1.81489795e-01,2.76485503e-01,5.21081537e-02,-1.01329692e-01,
-1.38789773e-01,1.24811279e-02,-1.30698040e-01,-7.87962005e-02,2.32123941e-01,
2.18225479e-01,-5.36812134e-02,-5.58626987e-02,3.22261333e-01,1.62614081e-02,
3.80255222e-01,-6.70613116e-03,5.36542945e-02,2.97504682e-02,-2.58912873e-02,
-6.73417971e-02,-6.95632026e-02,2.56226510e-01,1.67516544e-01,7.51638561e-02,
-3.21691781e-02,9.94723872e-04,-2.01429546e-01,1.16219949e-02,1.90720509e-03,
3.79896499e-02,1.57781497e-01,-6.73565641e-02,-1.71514526e-02,-1.98592082e-01,
9.77021307e-02,-6.75991103e-02,-1.78786978e-01,8.28748494e-02,-1.43360198e-01,
-7.53685236e-02,-1.67579755e-01,-4.66197617e-02,-7.41361529e-02,8.11314955e-02,
-8.80183131e-02,-2.05611199e-01,-1.72072634e-01,-5.85924760e-02,1.65178012e-02,
2.47013122e-02,1.09084420e-01,-2.86769290e-02,-4.42200750e-01,-1.40271634e-01,
3.45950335e-01,6.85522705e-02,4.33845483e-02,-2.03446113e-02,4.79266010e-02,
-1.54403578e-02,1.61066893e-02,-6.04595616e-02,3.93516794e-02,3.53031531e-02,
-9.93444864e-03,-1.64238624e-02,1.15612701e-01,-1.63441226e-01,-8.36442038e-02,
-4.59422544e-02,-4.18336689e-02,-7.24500045e-02,-1.71500221e-01,-4.34729494e-02,
-9.33034439e-03,1.87667623e-01,1.54777691e-02,-1.77747011e-01,-2.07989272e-02,
1.13116071e-01,7.79400244e-02,-1.29407346e-01,-1.19340964e-01,-1.66321114e-01,
-2.00685114e-02,1.26465753e-01,1.40083268e-01,-3.98499556e-02,1.53097305e-02,
1.48347482e-01,5.71398512e-02,1.25893261e-02,-8.65500495e-02,3.42258951e-04,
1.65787622e-01,1.08538456e-02,1.94535121e-01,-2.67789602e-01,-2.83708066e-01,
-1.89946592e-01,3.26868296e-02,-8.61642882e-02,1.27244279e-01,3.69604900e-02,
2.34932303e-01,-3.65323037e-01,-1.04788039e-02,-1.25747815e-01,3.29844840e-02,
8.14453661e-02,-1.41624035e-03,8.21648091e-02,-2.88737088e-01,-2.65483141e-01,
-7.54401609e-02,-1.62806556e-01,-3.92759219e-02,1.50541052e-01,-6.13040365e-02,
1.82517007e-01,-2.40567684e-01,-1.14213243e-01,2.14721724e-01,-2.09926456e-01,
-1.22918943e-02,-1.50753826e-01,1.68702472e-02,-1.35732204e-01,1.21451646e-01,
-2.30265725e-02,3.73614915e-02,-3.43028158e-02,-2.26752348e-02,3.50981429e-02,
6.16231300e-02,9.68883932e-02,-2.14138646e-02,-6.80466508e-03,4.50001173e-02,
1.94505136e-02,1.80319682e-01,1.46524563e-01,-7.72204436e-03,6.58486634e-02,
-8.88919923e-03,-2.15997323e-01,-9.17007495e-03,1.31882802e-01,2.80883666e-02,
6.54880702e-02,1.22365497e-01,4.14968580e-01,2.99156815e-01,3.61525677e-02,
-1.00156657e-01,-3.46969888e-02,5.95323229e-03,-4.69311178e-02,-2.47418061e-01,
-8.36511701e-03,9.11969170e-02,2.58804932e-02,2.36160263e-01,3.09393532e-03,
-5.68371601e-02,-2.64186054e-01,2.99552053e-01,1.92835927e-01,1.92356892e-02,
1.83849439e-01,1.40478268e-01,-1.27186671e-01,7.51584582e-03,-3.42759714e-02,
1.87599003e-01,7.82281235e-02,-2.58641303e-01,2.20815048e-01,-1.29182925e-02,
-1.92916617e-01,9.62471366e-02,-1.99198246e-01,-8.79264250e-03,1.82735220e-01,
6.02679439e-02,2.35782459e-01,1.10175349e-01,3.69065292e-02,4.01204033e-03,
-1.35159478e-01,-9.60067436e-02,-2.75048077e-01,1.57052487e-01,-5.13975359e-02,
1.33272782e-01,-1.03977844e-01,-5.45710512e-02,-2.79646128e-01,2.88026214e-01,
1.29067048e-01,-1.55875251e-01,-1.79343391e-02,2.50120256e-02,-1.48199230e-01,
-1.89522162e-01,2.13411655e-02,1.11263268e-01,3.69818836e-01,-4.99372371e-02,
-1.97849706e-01,-1.81424037e-01,8.91174376e-02,-1.96244985e-01,-8.63344520e-02,
-5.44225750e-03,-2.04959452e-01,-1.32062221e-02,3.99048418e-01,1.31682396e-01,
5.09787165e-02,-9.49388519e-02,2.68465698e-01,3.64847988e-01,1.50363043e-01,
-1.48433298e-01,-3.68272178e-02,1.17804267e-01,1.07638150e-01,1.67387500e-01,
-1.31701723e-01,-6.48216233e-02,-1.17614605e-01,1.91057414e-01,2.24860296e-01,
1.76276475e-01,6.02243468e-02,1.73206657e-01,1.37347341e-01,2.08163969e-02,
1.88033506e-01,2.18621731e-01,4.44360189e-02,2.88538188e-02,1.08291544e-01,
-1.00467466e-01,1.76001295e-01,-6.45302907e-02,6.17729165e-02,-1.22505561e-01,
9.83853824e-03,5.10559797e-01,4.71918434e-01,4.85921085e-01,2.08400879e-02,
-2.51394808e-02,-5.37012191e-03,-6.60763755e-02,-1.75356597e-01,1.39513612e-01,
-9.77202877e-02,-1.15217865e-01,-5.63748665e-02,-3.24190468e-01,5.08015528e-02,
-1.26200974e-01,-1.11516453e-02,-9.78174061e-02,1.41900733e-01,-3.49436045e-01,
-2.14945719e-01,-2.22089574e-01,-4.82581332e-02,1.73331499e-01,1.35533616e-01,
-5.39864413e-03,3.90208550e-02,1.52094454e-01,1.08706459e-01,3.74796480e-01,
-1.63788553e-02,1.77209139e-01,2.76332982e-02,1.12803027e-01,-1.41434774e-01,
2.61087865e-01,8.74700189e-01,1.66594665e-02,6.64546639e-02,4.26769704e-02,
-9.15345103e-02,1.50463656e-01,5.32628642e-03,2.48797357e-01,-8.30847397e-02,
2.31303275e-01,2.17192248e-01,-1.45561071e-02,2.38995418e-01,1.33429095e-01,
-9.00001377e-02,-1.12069942e-01,1.82891846e-01,1.78658925e-02,-1.19944677e-01,
2.84212559e-01,-2.02910870e-01,1.21532634e-01,5.47774844e-02,2.40634099e-01,
1.46956116e-01,1.39742985e-01,7.95922130e-02,1.24620572e-01,8.50849822e-02,
-2.74491072e-01,2.94898331e-01,4.25111949e-01,6.70706527e-03,4.16212767e-01,
4.55046855e-02,-1.83852509e-01,5.81203938e-01,3.05577815e-01,-2.83504766e-03,
2.88834460e-02,2.03249380e-01,-3.68238576e-02,5.25995851e-01,4.31228280e-02,
1.06571831e-01,1.26672640e-01,-2.13379979e-01,4.19998243e-02,-1.54357869e-02,
-3.84857468e-02,1.82602480e-01,-1.38389552e-02,-3.54246646e-01,1.94474906e-01,
-1.08801365e-01,-4.76336293e-02,3.77955511e-02,5.73559180e-02,-3.12813073e-01,
1.01916082e-01,-7.92242363e-02,-4.41352278e-02,5.80340736e-02,3.34964573e-01,
-3.84777009e-01,-5.28168231e-02,-2.45228529e-01,-7.57358316e-03,2.27322027e-01,
2.15920154e-02,-3.19015622e-01,5.60389683e-02,-1.30478039e-01,1.27890527e-01,
9.00472999e-02,8.38166773e-02,1.71002299e-01,-2.03827038e-01,2.40330279e-01,
-4.72751409e-02,4.74444404e-02,-6.70027966e-03,2.66935140e-01,-2.23506745e-02,
1.89758807e-01,2.27656737e-01,-2.56176125e-02,-5.96140884e-02,-6.08255751e-02,
5.53756207e-02,5.30987903e-02,-1.14349753e-01,-6.19521439e-02,-5.34660742e-02,
1.04209900e-01,-5.81451356e-02,-2.33237520e-01,2.98855513e-01,-6.71325698e-02,
-1.16215304e-01,-5.96586615e-02,-2.30548903e-02,1.62417039e-01,-1.25198603e-01,
-6.73936531e-02,9.50844213e-03,-4.09712940e-02,-7.38304928e-02,1.78579465e-01,
8.90375674e-02,3.70100178e-02,1.65704906e-01,-9.66042355e-02,2.04374686e-01,
-5.83223300e-03,1.69144630e-01,-5.58099337e-03,1.30088344e-01,2.05285251e-01,
2.12568771e-02,3.38594437e-01,5.30407131e-02,-7.29567036e-02,1.85137555e-01,
2.71295518e-01,2.63862252e-01,4.14278805e-02,-2.34167993e-01,2.18789987e-02,
2.39501670e-01,-2.88353004e-02,-1.75254643e-02,2.41047516e-02,6.52125850e-02,
-5.84360659e-02,2.25920066e-01,1.50989175e-01,4.81586382e-02,1.40536159e-01,
1.97257534e-01,2.05674861e-02,1.88271761e-01,-1.12792820e-01,2.00674042e-01,
-3.26378375e-01,2.63264507e-01,2.10706368e-02,-4.22099456e-02,-2.77357012e-01,
1.94742545e-01,-4.48674485e-02,-9.94991511e-02,2.30514124e-01,9.18590743e-03,
1.13719841e-02,1.21809080e-01,-1.95897132e-01,-4.74108681e-02,1.14854448e-01,
1.45402446e-01,5.43107353e-02,-1.17370687e-01,2.70170835e-03,-2.18649596e-01,
-1.33653864e-01,1.01741180e-01,7.79928938e-02,-2.80936826e-02,-1.00061834e-01,
-2.19254896e-01,-1.48242861e-01,-2.60858960e-03,1.56966165e-01,4.83550392e-02,
1.78947702e-01,2.23719236e-02,-7.21713379e-02,3.05623058e-02,9.67518985e-02,
1.20526083e-01,6.59040138e-02,-8.76054317e-02,1.41910419e-01,-1.90964028e-01,
2.19479352e-01,1.73095882e-01,1.61063358e-01,8.64659920e-02,-1.70906603e-01,
1.63080066e-01,1.06957771e-01,3.08265954e-01,-2.02290900e-02,-1.38711855e-01,
1.34969249e-01,7.46659935e-02,-1.20886318e-01,3.27135861e-01,2.51879036e-01,
-1.29581988e-01,3.25861387e-02,9.78865847e-02,-2.14990973e-01,8.54022652e-02,
-1.15606502e-01,6.21022061e-02,1.24879971e-01,2.20152736e-01,-2.89223462e-01,
-3.27521637e-02,-2.07976878e-01,-6.74783364e-02,3.34890723e-01,-1.61646053e-01,
1.07206315e-01,1.19149119e-01,-7.02220872e-02,1.20850295e-01,-2.63386965e-01,
1.69094220e-01,-2.98674524e-01,2.15155736e-01,-3.43895555e-01,-1.91158518e-01,
-3.86866927e-01,-3.47024232e-01,-6.99302405e-02,-1.55278906e-01,3.50137860e-01,
-6.97622970e-02,1.64034367e-02,-5.20423986e-02,-1.57463267e-01,4.85633314e-02,
3.57127353e-03,-1.65096223e-01,-2.27721855e-01,4.71206978e-02,6.93155900e-02,
-1.09087639e-01,1.46899879e-01,1.23108871e-01,-6.63804486e-02,1.33000612e-01,
-3.20968293e-02,2.49472656e-03,-1.96488183e-02,-1.15658879e-01,-1.01489677e-05,
-6.73692971e-02,-1.62916239e-02,-1.83619857e-02,9.00513772e-03,-1.03853434e-01,
1.03205673e-01,-1.75087512e-01,1.38749316e-01,1.53297007e-01,7.08645508e-02,
-2.23742127e-02,8.82896855e-02,1.92058340e-01,-1.14729002e-01,-6.51037768e-02,
-5.41114025e-02,-4.58617089e-03,-1.03013821e-01,-6.25739917e-02,1.37731075e-01,
9.40585956e-02,1.01630896e-01,-1.23732530e-01,-4.10679029e-03,-2.12494537e-01,
-2.00768203e-01,-1.53431788e-01,-2.34752119e-01,-1.36620402e-01,-1.71490148e-01,
-5.80068156e-02,-7.14449659e-02,-5.11221401e-02,-2.21588165e-01,-1.20456882e-01,
-2.13423535e-01,5.85382432e-02,-7.51645342e-02,-1.19872339e-01,1.45109251e-01,
-3.45792174e-01,-3.34436864e-01,1.83080211e-02,-2.19702750e-01,-2.94293985e-02,
-8.14263374e-02,-2.21562937e-01,6.00091033e-02,-1.27905101e-01,-1.69410691e-01,
-1.56516075e-01,-3.85867685e-01,1.30577058e-01,-8.30358788e-02,1.04975417e-01,
-1.47143647e-01,6.98385537e-02,-5.57604134e-02,-1.87060647e-02,-1.01886153e-01,
5.42390086e-02,-1.58133999e-01,1.84540778e-01,-1.14953287e-01,-3.81611213e-02,
9.59815234e-02,-4.03191261e-02,1.11379743e-01,-1.12339325e-01,1.70729798e-03,
2.00684555e-02,6.48544952e-02,7.80407861e-02,-6.20913729e-02,1.35069549e-01,
-8.55774209e-02,1.84224293e-01,5.29961213e-02,-3.03761326e-02,5.50387390e-02,
-1.84412617e-02,1.14138238e-01,-2.04285122e-02,5.00655100e-02,9.68416780e-02,
-1.62331253e-01,1.12568997e-01,-1.18363269e-01,-1.21652417e-01,-5.14121428e-02,
-2.12303009e-02,-3.98042798e-02,-2.19840825e-01,-1.15310989e-01,-6.97613508e-02,
-5.53379580e-02,-1.96386561e-01,-2.17082277e-01,4.74696048e-02,-1.18615381e-01,
-2.76537865e-01,7.61682913e-02,2.00536405e-03,-1.63775936e-01,-2.44812146e-01,
-1.83608666e-01,-3.13338161e-01,-4.53001380e-01,-4.74431925e-02,-7.07378164e-02,
-2.69969553e-01,-1.30163617e-02,-3.22911814e-02,-1.74625758e-02,-1.17473252e-01,
-3.03869545e-01,-2.26495266e-01,-1.60657287e-01,-2.11999312e-01,8.30724090e-02,
-6.24202453e-02,-1.27359003e-01,-9.13780835e-03,2.53287792e-01,-5.97677454e-02,
-1.07299574e-01,4.33269106e-02,-9.69311073e-02,-6.62497282e-02,-2.18744809e-03,
1.02933556e-01,-1.00405797e-01,5.05795740e-02,-6.87870085e-02,1.08026244e-01,
-1.16521746e-01,-2.29094923e-01,-7.97517132e-03,1.05493724e-01,-1.24664074e-02,
4.46161330e-02,-1.09770559e-01,-1.39568299e-02,-2.11520270e-01,-2.90430058e-03,
-1.25579452e-02,4.62923236e-02,2.51330733e-02,-1.55706972e-01,-2.18148082e-01,
1.32823130e-03,6.94403201e-02,1.84048444e-01,3.42436954e-02,-1.82453334e-01,
-1.41471243e-02,3.75195965e-02,5.92083000e-02,-3.04945827e-01,-3.27531457e-01,
3.82410809e-02,1.02012560e-01,-1.73064291e-01,-9.94858816e-02,-4.38161381e-02,
-1.09568588e-01,-1.45425528e-01,-3.29281062e-01,-3.65481414e-02,-4.59393347e-03,
1.12846769e-01,-1.97300702e-01,-2.38777086e-01,-2.82109380e-01,1.48013636e-01,
-1.21693909e-01,2.18319133e-01,-9.26097482e-02,2.25447938e-01,1.48278996e-01,
-4.60367620e-01,-6.24430180e-02,-1.43913403e-01,-2.51143575e-01,4.15804476e-01,
2.25226488e-02,-6.40531853e-02,-1.80829354e-02,2.94725560e-02,1.93249865e-03,
8.59665275e-02,-1.19643316e-01,-3.45311798e-02,6.69920594e-02,7.14888349e-02,
2.09480301e-01,-3.87059301e-02,-2.53599789e-02,1.81718513e-01,-2.42934808e-01,
1.18085491e-02,4.35506403e-02,-1.39492542e-01,5.42626642e-02,1.14688054e-01,
-1.50486410e-01,2.44194437e-02,6.82587698e-02,-8.22360888e-02,-1.55419791e-02,
-2.38053456e-01,1.97097454e-02,2.14718401e-01,4.43068705e-02,-6.85110316e-02,
4.57439013e-02,-2.36934513e-01,7.19281435e-02,5.32082580e-02,3.38735804e-02,
7.35181347e-02,2.32445393e-02,1.68071941e-01,3.83974798e-02,-1.01704441e-01,
3.19660783e-01,3.54885273e-02,-4.48231213e-02,-2.04198033e-01,2.19004348e-01,
-3.64736989e-02,-1.92329586e-01,-3.84008437e-02,-1.64058700e-01,-1.21670114e-02,
-2.12054513e-03,5.19136749e-02,-1.06138110e-01,7.66051486e-02,-1.56061664e-01,
-2.77657621e-02,1.18388243e-01,1.95257321e-01,4.77669947e-03,-2.14558206e-02,
-5.32960929e-02,-5.36663160e-02,5.30975237e-02,-3.36058661e-02,8.42018872e-02,
1.11878656e-01,1.01300456e-01,8.11115187e-03,-1.31540641e-01,2.72062495e-02,
-1.77902207e-02,-1.88890949e-01,-1.54160276e-01,2.77991388e-02,3.01817413e-02,
8.06114897e-02,-2.74050176e-01,-2.90729348e-02,-9.90870446e-02,-1.55680561e-02,
4.34079673e-03,-2.78918836e-02,-8.52196589e-02,-2.97778636e-01,2.47050356e-02,
-2.77640939e-01,-2.00369909e-01,-1.57225788e-01,-1.01483293e-01,9.74937156e-03,
1.21535599e-01,-3.28260958e-02,-1.58458925e-03,-2.34723896e-01,-3.28860357e-02,
-2.04566568e-01,-1.71022728e-01,-1.25074416e-01,-5.33631966e-02,-4.35854830e-02,
1.13689110e-01,-1.13496371e-01,-3.28226686e-02,2.35446673e-02,-1.14834912e-01,
-2.81201065e-01,-2.28072498e-02,-6.25143498e-02,-9.35553573e-03,-8.83862376e-02,
-2.93468446e-01,-1.57039031e-01,-1.54096350e-01,-6.63890094e-02,2.01174542e-02,
1.47596329e-01,-7.69519806e-02,-1.16345093e-01,-1.31154582e-01,-1.46460578e-01,
-2.43352935e-01,-3.08318257e-01,-7.97477644e-03,2.04637706e-01,-6.20619617e-02,
2.35688359e-01,1.10587291e-01,-2.73486346e-01,1.57015443e-01,-3.82495150e-02,
-2.82859385e-01,1.62857234e-01,2.35259399e-01,4.99827452e-02,3.68841514e-02,
4.16838340e-02,-1.81073863e-02,2.80404389e-02,2.33805448e-01,2.11140454e-01,
-1.46062687e-01,-3.63169275e-02,-1.25538513e-01,-6.71839342e-02,-3.85669112e-01,
-2.31020331e-01,-2.48231113e-01,-6.82034194e-02,-1.36897355e-01,-1.57969460e-01,
-1.67092562e-01,-1.43439006e-02,-3.16904843e-01,-3.07816505e-01,-4.16595712e-02,
-2.21006885e-01,-1.99885294e-01,-2.24208869e-02,-1.41808391e-01,5.07143512e-02,
-3.20258558e-01,1.26278838e-02,-1.63423523e-01,-3.26431662e-01,-1.17049411e-01,
1.95232481e-01,-1.30412191e-01,-6.13714494e-02,1.18718920e-02,-9.64345187e-02,
-3.86807173e-02,-1.92404985e-01,-1.15214624e-01,1.23505481e-01,-3.16283144e-02,
-2.67968234e-02,-2.00519085e-01,-1.91250890e-01,-1.55864626e-01,-9.16588977e-02,
-5.81351258e-02,6.19720705e-02,-1.47602603e-01,4.49573770e-02,-4.36204337e-02,
-1.12827353e-01,-1.80041730e-01,1.55077919e-01,5.45651056e-02,2.31871824e-03,
-1.08984545e-01,-1.13013931e-01,-2.29765046e-02,3.86970788e-02,4.89628911e-02,
-9.90089308e-03,-7.48828575e-02,2.02673092e-01,1.53451100e-01,-2.09699795e-02,
7.32320026e-02,2.90515497e-02,1.15943514e-01,-6.69980794e-02,-1.46091551e-01,
-6.08361699e-02,1.38214245e-01,7.84807578e-02,7.78899761e-04,-1.51167691e-01,
-1.81521803e-01,7.80269457e-03,-3.27147171e-02,6.25394285e-02,-1.80994987e-01,
8.71972889e-02,-6.91654757e-02,-2.20560968e-01,-1.10134155e-01,3.51930782e-02,
-9.99897942e-02,-7.05865771e-02,1.55456826e-01,-2.21715391e-01,-1.89060867e-01,
-1.37323856e-01,-2.26655424e-01,7.17517287e-02,-5.94461821e-02,-2.46099323e-01,
1.22323170e-01,-2.57104397e-01,-1.60841234e-02,-8.07608366e-02,-1.27584708e-03,
-1.73683345e-01,-6.86972961e-03,2.79990703e-01,-3.04760933e-01,-2.33475804e-01,
-3.64347398e-01,6.79353811e-03,-5.16338386e-02,-9.59078223e-02,3.60143781e-02,
-3.69438142e-01,-2.01118231e-01,-2.03332990e-01,-3.64798248e-01,-7.95048922e-02,
-8.37974548e-02,-4.93891507e-01,-2.58681953e-01,9.85702686e-03,-1.28697723e-01,
-8.70549753e-02,-3.24398190e-01,-3.25475708e-02,-4.64191586e-02,-3.85843605e-01,
-1.52269542e-01,-4.32103485e-01,-4.36883688e-01,-2.47541373e-03,-3.68343927e-02,
2.42451042e-01,-9.76226628e-02,1.03507660e-01,5.09102009e-02,2.32822895e-02,
-1.04166172e-01,1.17015451e-01,1.07538350e-01,2.89908350e-02,-9.41667631e-02,
-1.66667044e-01,-3.34841125e-02,-2.04882205e-01,-3.70945334e-01,-1.83087111e-01,
-1.02715582e-01,-3.35092247e-02,-2.83076763e-01,-2.04598621e-01,-4.28745151e-02,
-2.60519475e-01,-1.66706145e-01,-1.93228573e-01,1.57411061e-02,-3.71540397e-01,
-1.02562487e-01,-1.26545448e-02,-1.44182876e-01,2.77770218e-03,-8.43603760e-02,
2.18467321e-03,-4.95501757e-02,3.50901000e-02,8.40716511e-02,6.91063553e-02,
-3.52387875e-02,-1.13587258e-02,5.86749353e-02,-1.66892350e-01,2.38611564e-01,
-8.38882774e-02,6.13063052e-02,-1.37248278e-01,7.93535188e-02,-1.55736953e-02,
3.59013155e-02,-7.93498158e-02,-3.69387776e-01,-3.57321091e-02,1.69146154e-02,
6.59695417e-02,-1.52692184e-01,-1.24277063e-01,-1.87404245e-01,2.95628123e-02,
1.23447672e-01,3.88996303e-02,5.97716635e-03,-1.45444483e-01,-3.38448398e-02,
-1.68415472e-01,6.19991012e-02,-1.40728518e-01,-1.74923435e-01,1.78314671e-01,
-2.84328848e-01,-3.19235325e-02,7.83844441e-02,2.47747116e-02,2.59186268e-01,
-6.10715933e-02,-1.74025595e-01,2.67087221e-02,4.78288084e-02,1.97212607e-01,
-1.91871017e-01,-4.40730639e-02,-2.16433406e-02,-7.74865896e-02,-1.89345941e-01,
-5.74004203e-02,-9.44347903e-02,2.11325809e-01,-9.03477594e-02,-4.68190154e-03,
-1.54619992e-01,2.49352697e-02,-7.01622888e-02,-1.67528719e-01,9.51200575e-02,
-8.06942359e-02,1.03672966e-01,-4.50877100e-02,-1.15980275e-01,-1.15566365e-01,
-1.28666073e-01,4.73546050e-03,1.02837577e-01,3.37194018e-02,-2.22506329e-01,
-2.50252247e-01,-3.22307460e-02,-1.93300650e-01,-9.58147421e-02,5.82889728e-02,
-4.62118387e-02,3.30680385e-02,2.80135497e-02,-5.67800105e-02,-1.61651820e-01,
-2.81878471e-01,-2.18143433e-01,-2.06522033e-01,-2.14625493e-01,2.37622950e-03,
-2.28670597e-01,-1.68473631e-01,5.74741401e-02,-1.57494292e-01,-3.01808596e-01,
-4.56070229e-02,-4.59194891e-02,6.96549714e-02,-3.14198256e-01,-2.35053167e-01,
6.54225945e-02,-1.40355170e-01,-1.67425931e-01,-1.48000360e-01,-3.32837701e-01,
-1.28372729e-01,1.80351418e-02,-6.43660650e-02,2.16197465e-02,6.97242245e-02,
1.97938597e-03,-1.01895064e-01,2.35754028e-02,-1.26814410e-01,-1.86190143e-01,
-2.41907444e-02,-1.91515431e-01,4.07143217e-03,-1.09515913e-01,1.32155046e-02,
-4.69210558e-02,-1.36051491e-01,-1.78616092e-01,-1.20330323e-03,-3.78858782e-02,
-1.60691440e-01,-9.48938578e-02,-2.23668009e-01,2.69104373e-02,1.07009605e-01,
3.52719799e-02,8.29490572e-02,2.45592818e-01,-3.20828825e-01,4.96964678e-02,
1.33282885e-01,-2.10703731e-01,-6.76646233e-02,1.92464009e-01,-2.10798413e-01,
8.10979307e-02,-3.71323898e-02,-4.25264016e-02,2.05352142e-01,6.07781000e-02,
4.43393737e-02,-3.85594629e-02,-6.25632182e-02,-1.35077477e-01,-1.40854329e-01,
-1.39157906e-01,-1.08387515e-01,3.35281417e-02,-2.38602728e-01,-9.61619243e-03,
-1.38877869e-01,-5.50543889e-02,-2.03982159e-01,-2.74123177e-02,-9.00816824e-03,
-4.90056761e-02,-5.01290560e-02,-2.28865743e-01,-1.88623503e-01,-1.59469768e-01,
-2.98490882e-01,-5.07586375e-02,-3.34168911e-01,-2.17350185e-01,-7.19408244e-02,
-5.55709749e-02,-9.44213867e-02,-1.63457572e-01,-1.44503936e-01,-1.06098995e-01,
5.24908490e-02,1.74435765e-01,-1.40169889e-01,-2.99974740e-01,-1.36587411e-01,
-4.94798599e-03,-1.61807477e-01,-1.46571115e-01,-6.02431931e-02,-1.79917924e-02,
-2.13182479e-01,7.61430487e-02,-2.46706456e-01,-9.30184778e-03,-2.22165808e-02,
-1.12340666e-01,-1.23258621e-01,-1.78259924e-01,-3.06397140e-01,1.04721989e-02,
1.10063486e-01,6.52368888e-02,-6.84137195e-02,1.24898694e-01,-1.62401255e-02,
-5.40965021e-01,-2.30595823e-02,1.23172544e-01,-4.38382536e-01,2.78402925e-01,
3.76928821e-02,-2.43483968e-02,-2.75849103e-04,4.48872306e-04,-3.49470004e-02,
-2.93535799e-01,4.14529853e-02,-1.61321953e-01,1.64470345e-01,1.71841949e-01,
-7.19738826e-02,-1.01729393e-01,-2.08658919e-01,1.73748448e-01,-2.82091111e-01,
6.70484528e-02,1.59202248e-01,-2.37818360e-02,-2.07124904e-01,-5.96743785e-02,
-1.47102281e-01,2.90692508e-01,-6.76655695e-02,-2.86906515e-03,4.60877270e-02,
-1.67535812e-01,2.31063366e-02,5.63854948e-02,-7.19354525e-02,6.78158849e-02,
1.06944854e-03,-3.43549661e-02,7.70172775e-02,3.47043425e-02,-7.94882923e-02,
-5.82834333e-02,8.65515992e-02,-2.31542997e-02,6.57636076e-02,-1.44423932e-01,
2.18635369e-02,1.22173287e-01,1.80900782e-01,1.02970243e-01,4.14995737e-02,
1.24542229e-01,-1.09584935e-01,-2.09269971e-01,-1.47142872e-01,1.99550956e-01,
-1.04533527e-02,-2.67345339e-01,1.47911049e-02,-1.71585470e-01,-2.23618895e-01,
-2.47915685e-01,-3.78830612e-01,2.46900380e-01,-6.79805577e-02,-1.03602493e-02,
2.51463354e-01,2.54132301e-01,-8.30972493e-02,-4.16192636e-02,-6.38079504e-03,
-1.08911626e-01,-2.18913585e-01,1.48028238e-02,-7.57999793e-02,3.81343178e-02,
-2.10612297e-01,-2.02223763e-01,9.47828144e-02,-3.17082144e-02,7.94048756e-02,
-1.07970484e-01,1.63391959e-02,-3.23705226e-01,-3.01675797e-01,-1.99574843e-01,
-4.80276078e-01,-1.22577660e-01,-9.23167616e-02,-1.34320930e-01,4.92975749e-02,
-3.89601171e-01,-6.39162123e-01,-4.24345762e-01,-1.99719686e-02,-1.13646373e-01,
-1.44672975e-01,3.07936836e-02,3.71263959e-02,5.48123978e-02,-9.11836606e-03,
-1.18484154e-01,-2.74511784e-01,-6.39878213e-02,9.23218131e-02,2.91572399e-02,
1.00688793e-01,6.86567798e-02,-2.76722806e-03,-1.89252213e-01,-1.38905838e-01,
-1.63384378e-01,1.82498083e-01,2.32553661e-01,2.43499354e-01,-1.49000928e-01,
1.05515406e-01,-1.05869353e-01,-1.20859481e-01,-4.50004905e-01,-4.41782735e-02,
5.43295443e-02,-1.20351173e-01,-2.14290187e-01,-1.49135679e-01,-5.47536373e-01,
-3.91223490e-01,-4.71910015e-02,-2.50506401e-01,-1.58280402e-01,-3.69300634e-01,
4.75611202e-02,3.85794163e-01,6.49765972e-03,-3.84574980e-01,-1.45751894e-01,
-2.23267451e-01,-2.49629319e-01,6.50654584e-02,7.02168653e-03,-1.56997621e-01,
-1.63073286e-01,-4.01349366e-01,5.94301559e-02,-4.97822352e-02,1.36291357e-02,
-5.75205311e-02,8.00818112e-03,8.33771750e-02,-2.17985272e-01,-3.44325572e-01,
-1.15884617e-01,-1.55750647e-01,3.67144980e-02,1.74012389e-02,-1.81818739e-01,
-3.04442316e-01,2.65478576e-03,-3.00366819e-01,-6.07449532e-01,-2.14008436e-01,
-3.76506709e-02,-1.03533112e-01,-2.50676662e-01,-8.53146985e-02,4.23440374e-02,
8.71740729e-02,8.64959806e-02,-1.54554024e-01,-1.61880553e-01,-2.95078188e-01,
5.22702886e-03,-3.29838395e-02,1.15714908e-01,-1.27533302e-01,1.07694805e-01,
-7.67657114e-03,-1.28911227e-01,3.91641930e-02,6.46699406e-03,-3.23807448e-02,
-1.70489684e-01,2.26088047e-01,-1.15344152e-02,-9.45342332e-02,2.34238267e-01,
-1.80564180e-01,-1.60686165e-01,5.80615029e-02,2.00950682e-01,-2.00665556e-02,
1.12994537e-01,-2.31512934e-02,2.68862307e-01,4.54024896e-02,-1.25169000e-02,
2.26885960e-01,2.97384173e-01,2.50712447e-02,-2.02002540e-01,1.07688360e-01,
-1.37059018e-01,2.39383787e-01,1.80186808e-01,-1.06617540e-01,6.26165867e-02,
7.94074237e-02,6.26681298e-02,9.98343676e-02,8.70300382e-02,2.84486078e-02,
2.23829120e-01,6.80925772e-02,7.92605430e-02,2.47345239e-01,-6.06491347e-04,
-1.37782678e-01,2.70786546e-02,-3.88613641e-02,3.01446170e-01,1.35160834e-01,
-3.66396680e-02,4.63707261e-02,-1.10937856e-01,-6.36318624e-02,7.23440647e-02,
-1.46117583e-01,1.52752995e-01,3.48650485e-01,-7.68596604e-02,1.19339772e-01,
-8.43180344e-02,-1.51078939e-01,-2.81791985e-01,1.27387986e-01,-6.37978036e-03,
3.36067051e-01,-2.60368157e-02,-4.33710031e-02,-3.82868126e-02,3.57959941e-02,
1.42508760e-01,3.15726697e-01,-1.64103676e-02,4.43207547e-02,5.97895384e-02,
-1.26931444e-01,2.57489476e-02,1.99135676e-01,-1.72095131e-02,-9.22816247e-02,
2.78304350e-02,9.25309882e-02,1.14135765e-01,1.61871001e-01,-1.86995581e-01,
-1.53196976e-01,1.56735793e-01,-3.50569561e-02,-7.08520040e-02,3.87085043e-02,
9.70567465e-02,-1.86178729e-01,3.06194695e-03,-3.35736875e-03,1.58434018e-01,
-3.31280857e-01,-4.57902551e-02,1.05098978e-01,-4.04401049e-02,-2.67540634e-01,
3.60121690e-02,-1.87962875e-01,-1.35684116e-02,3.42446715e-02,6.90284893e-02,
2.34298632e-02,5.81421852e-02,4.53477986e-02,1.46597112e-03,1.06196009e-01,
1.58216104e-01,-8.70073512e-02,-5.53788804e-02,8.82656574e-02,-7.13615417e-02,
1.53019413e-01,1.29267454e-01,3.92984748e-02,1.55888066e-01,2.11727545e-01,
5.11204414e-02,1.09649912e-01,-1.93385050e-01,8.66358913e-03,-2.55039125e-03,
-5.32190278e-02,2.35180676e-01,-2.00993940e-01,-5.59225418e-02,1.46335781e-01,
5.01085073e-02,3.18398714e-01,4.90619279e-02,-1.28601834e-01,5.83970314e-03,
1.08268335e-01,-1.39069229e-01,2.64514312e-02,1.08724862e-01,8.27106163e-02,
-2.03223228e-02,2.47083697e-02,1.60345994e-02,-4.69457619e-02,7.30442405e-02,
-2.25563467e-01,-7.52061307e-02,1.92019660e-02,3.12661618e-01,-5.36582842e-02,
1.74124569e-01,-2.27022424e-01,2.50600874e-02,-2.46864706e-01,1.39668494e-01,
6.81246072e-02,1.23780154e-01,-1.26321152e-01,1.42137915e-01,-1.06756449e-01,
-2.50984967e-01,2.52732839e-02,-9.87740532e-02,-5.17159939e-01,2.93572545e-01,
-7.09356889e-02,-5.38347512e-02,-3.44375707e-02,-5.36028156e-03,-2.28555929e-02,
-6.25269953e-03,4.18245718e-02,8.79327953e-02,2.75495321e-01,1.56213850e-01,
-7.21851960e-02,1.41866086e-02,4.15474027e-02,1.77899655e-02,1.92246325e-02,
1.57266650e-02,1.36419117e-01,-9.62054506e-02,-3.46731216e-01,6.00931644e-02,
-6.41158521e-02,-5.28922491e-02,-1.27970383e-01,-1.54049248e-02,3.68930489e-01,
-2.09672004e-01,-1.59014225e-01,2.90398300e-01,-2.69734301e-02,-2.43085429e-01,
-1.88967034e-01,-3.81065384e-02,-2.81980276e-01,4.10473913e-01,-9.55172628e-02,
6.12379424e-02,-1.11208484e-03,-6.74001593e-03,-4.16654870e-02,1.49335757e-01,
-1.95577458e-01,-2.15302482e-01,1.51256308e-01,8.27330947e-02,3.05414144e-02,
-1.06596174e-02,-9.79039297e-02,1.09524034e-01,2.51599960e-02,-9.70694423e-02,
-9.24292877e-02,-7.61052640e-03,2.33202297e-02,-8.19539744e-03,-1.45417899e-01,
-3.69015783e-02,-2.93855146e-02,2.39305794e-01,1.02465317e-01,1.12273164e-01,
-7.57701620e-02,2.38369897e-01,2.51641050e-02,2.78317720e-01,2.81661272e-01,
9.80019942e-02,-4.15491998e-01,3.54416557e-02,5.75740874e-01,1.57247603e-01,
5.61893359e-02,-2.83587515e-01,6.15109392e-02,-1.08897164e-01,-3.63087691e-02,
6.30286112e-02,4.21665870e-02,7.83437304e-03,-6.59942776e-02,-1.42620310e-01,
1.10806182e-01,-7.25872740e-02,-7.68432841e-02,-8.81001130e-02,5.27156703e-02,
1.88383348e-02,-1.28563881e-01,1.28535330e-01,-1.64586194e-02,2.15488281e-02,
2.84151584e-01,1.17986403e-01,2.49847677e-02,-6.69026822e-02,1.26731232e-01,
-2.14704294e-02,-8.32494944e-02,6.05797209e-02,2.47643694e-01,1.98551327e-01,
-8.41502100e-02,-1.71344772e-01,-8.33606198e-02,1.17182948e-01,-2.02315927e-01,
4.80378531e-02,5.67582883e-02,-3.04747611e-01,-8.11739266e-02,-2.64214594e-02,
-1.32516637e-01,1.38507724e-01,8.75275582e-02,1.54274881e-01,-1.65323392e-01,
-8.20738822e-03,-2.23813877e-01,-6.15030387e-03,1.78366855e-01,-2.03386217e-01,
-1.36388004e-01,1.37586668e-01,7.62932003e-02,4.03178722e-01,2.27133706e-01,
3.26515466e-01,-3.66808362e-02,-8.38731080e-02,-1.38770565e-01,-9.84939635e-02,
3.04497685e-02,-1.08054422e-01,4.06921841e-02,-4.87887114e-02,-6.26306534e-02,
-4.49196510e-02,8.68999138e-02,-1.81731917e-02,3.91379111e-02,-2.89581437e-02,
6.63736090e-02,-9.89637338e-03,1.65520739e-02,1.46969166e-02,-1.00839771e-01,
-1.21410020e-01,1.81939849e-03,7.97497928e-02,1.23395771e-02,-1.30048484e-01,
9.82278436e-02,1.59112699e-02,2.06303187e-02,-9.36860815e-02,-7.72596244e-03,
-6.74629360e-02,-4.73048501e-02,6.59830356e-03,-1.19396701e-01,-1.92673430e-02,
1.64765790e-01,-2.15967864e-01,7.41685368e-03,7.65561238e-02,-3.78302634e-02,
2.25614130e-01,-2.90886104e-01,-1.73801079e-01,-3.75412311e-03,2.03118622e-01,
1.50193930e-01,-6.14895299e-02,2.71120131e-01,2.14410365e-01,4.65603173e-03,
1.80456042e-01,9.61597711e-02,2.32043881e-02,-7.82792866e-02,-3.49741966e-01,
-1.91815645e-01,4.18318138e-02,-2.67029762e-01,5.84820881e-02,-1.31123187e-03,
8.86305571e-02,1.53853430e-03,6.67448789e-02,-3.97948325e-02,-5.98805696e-02,
1.54196128e-01,3.06925774e-01,1.25219584e-01,-5.79779521e-02,2.08059009e-02,
-1.10491753e-01,1.08704053e-01,-1.45645104e-02,4.48079258e-01,-1.43076330e-01,
-9.04373974e-02,1.23482637e-01,-1.24862000e-01,6.61363676e-02,1.94752857e-01,
7.08684251e-02,-2.67271772e-02,-1.84491444e-02,-9.54130944e-03,-1.10398702e-01,
8.22565481e-02,1.25419823e-02,1.13460563e-01,1.01310357e-01,1.12722710e-01,
-2.59820133e-01,-5.05242534e-02,6.01276904e-02,1.52083412e-02,2.31994808e-01,
5.53059839e-02,6.07533194e-02,-1.42231748e-01,1.23745747e-01,-2.72020306e-02,
1.50719970e-01,3.24767977e-01,-5.38968928e-02,6.63863942e-02,3.94138008e-01,
-1.41649678e-01,5.22591695e-02,-1.37554377e-01,6.80541396e-02,-9.52280089e-02,
-1.93643019e-01,-4.51206900e-02,-1.49554759e-01,-2.27188338e-02,-1.07713521e-01,
2.77869910e-01,-3.19762051e-01,-1.76173583e-01,2.53473427e-02,-4.14320558e-01,
-1.29902810e-01,-6.78815722e-01,-2.29788989e-01,2.32463822e-01,-7.68899471e-02,
-1.37681896e-02,-2.41158903e-01,-3.77627045e-01,-1.51839599e-01,2.66120523e-01,
-6.70705914e-01,4.45008092e-02,1.85026601e-01,-1.77753136e-01,-3.24355721e-01,
-2.33226065e-02,-1.95928648e-01,-2.42594540e-01,1.44612491e-01,4.76883560e-01,
-2.71880984e-01,1.15502112e-01,-1.06597476e-01,7.66538270e-03,2.21513420e-01,
-1.26970336e-02,2.07567781e-01,-5.93646392e-02,-6.13251030e-02,3.45144242e-01,
1.46703944e-02,-6.01716973e-02,6.92082196e-02,-3.52023891e-03,3.64847332e-01,
4.93118390e-02,-9.76322815e-02,3.90958309e-01,1.55986324e-01,-1.03488430e-01,
3.99940729e-01,2.13495076e-01,2.90929794e-01,-1.47602051e-01,-6.23118803e-02,
-1.79103002e-01,2.26062492e-01,5.84532678e-01,-3.06547880e-02,-2.29579024e-02,
5.41156888e-01,-6.51856791e-03,5.31197608e-01,-4.03705277e-02,3.79906923e-01,
-4.72937077e-02,-9.69978869e-02,2.79971004e-01,-2.41220340e-01,5.86080015e-01,
1.18785128e-01,2.38681495e-01,-8.75244811e-02,5.82025498e-02,4.25967544e-01,
-1.90061703e-01,-3.26066203e-02,2.06984431e-01,4.78433669e-01,5.91296963e-02,
-3.42249461e-02,2.45723724e-01,3.33971024e-01,9.73085761e-02,1.18764684e-01,
-1.41491547e-01,5.07607758e-01,3.23048264e-01,1.19593874e-01,2.07378075e-01,
2.07008526e-01,1.08088776e-01,4.23128575e-01,2.80500323e-01,6.93338513e-02,
3.05758506e-01,-4.43609841e-02,2.20244259e-01,8.02297965e-02,2.10271433e-01,
4.70782109e-02,-1.57526746e-01,-1.78435773e-01,5.69182113e-02,5.69227301e-02,
-8.84478167e-02,-3.79030928e-02,-9.73828719e-04,-6.60044104e-02,8.26927423e-02,
-1.44743487e-01,2.54561871e-01,-6.81937411e-02,2.87093371e-02,2.09580228e-01,
-8.59868079e-02,6.17589802e-02,-2.60082990e-01,-2.59030581e-01,6.98676184e-02,
1.98245600e-01,1.34750247e-01,1.43273726e-01,1.92247659e-01,-1.14526786e-01,
3.50747705e-02,3.24203104e-01,5.03904969e-02,1.07760400e-01,5.15136449e-03,
-3.32564376e-02,1.37841359e-01,-9.56307072e-03,-1.21614695e-01,-1.99028514e-02,
-3.59074473e-02,-3.65937620e-01,-3.82130355e-01,-1.63474798e-01,-1.59450546e-02,
-1.69842809e-01,-9.48579982e-02,-2.32201070e-01,-2.13884562e-01,-1.92465439e-01,
2.50710249e-01,7.63573498e-02,-1.84947580e-01,-1.10072747e-01,-1.00012243e-01,
1.05555959e-01,3.59165110e-02,-4.22033489e-01,3.85927223e-02,6.54414743e-02,
-1.42039195e-01,-3.14405501e-01,2.78099743e-03,1.20893857e-02,2.80653741e-02,
2.08691850e-01,3.85375351e-01,1.93047911e-01,9.61841345e-02,-1.88876670e-02,
2.22307933e-03,3.07709485e-01,-2.34191105e-01,3.33990574e-01,1.89541996e-01,
1.51587009e-01,2.50403494e-01,7.51855373e-02,-2.32023764e-02,-4.92149182e-02,
5.14660776e-02,-2.49225676e-01,-3.53169292e-02,-4.00134251e-02,1.19745456e-01,
3.17158669e-01,-7.64441788e-02,8.44881162e-02,-9.25093517e-02,-7.42060319e-02,
2.21738890e-01,3.36290240e-01,4.54515398e-01,1.61969632e-01,-8.83182418e-03,
-2.93071549e-02,-3.97286236e-01,5.47009945e-01,1.93208307e-01,2.23348036e-01,
-2.19845042e-01,9.70852301e-02,-1.79525152e-01,-3.29915762e-01,-1.46427765e-01,
8.98964405e-02,2.79023685e-02,-4.06514779e-02,5.55087961e-02,-1.23280371e-02,
-3.04494519e-02,2.68852655e-02,1.54712036e-01,1.77311033e-01,-5.30978069e-02,
1.95944786e-01,-1.56030029e-01,2.34297365e-01,2.78923362e-01,3.06794554e-01,
5.08112609e-02,6.07067868e-02,3.26269269e-02,3.61509435e-02,1.96110964e-01,
3.05039287e-01,1.32149979e-02,9.19650719e-02,-2.03242153e-02,5.13449721e-02,
6.57104254e-02,-7.11076781e-02,5.58474250e-02,4.13988717e-02,2.24431269e-02,
3.79818566e-02,-1.11340217e-01,-3.87271553e-01,-4.97697778e-02,-1.33311555e-01,
-2.82456100e-01,-2.67454177e-01,-1.17102996e-01,-8.08897614e-02,-4.67825472e-01,
-1.89512521e-02,-2.71797091e-01,-4.16681856e-01,-3.44860107e-01,1.60035223e-01,
-1.83931477e-02,-5.42655230e-01,-6.04289472e-01,-1.03287827e-02,-6.32964194e-01,
1.40253544e-01,-1.12831488e-01,-4.18735474e-01,-3.87823254e-01,-2.93224484e-01,
-4.34832394e-01,3.36628184e-02,1.04495332e-01,-3.36758822e-01,-9.43396688e-02,
-2.54746497e-01,-2.35179216e-01,-1.39472112e-01,1.22776173e-01,-2.16387302e-01,
-3.32515389e-01,5.91613233e-01,-1.89069435e-01,-2.47709870e-01,-3.07417095e-01,
2.16346651e-01,-2.07126647e-01,-3.20175499e-01,2.05866635e-01,3.93833704e-02,
9.79862362e-02,-4.47500110e-01,-3.19327384e-01,3.94946011e-03,-2.83864349e-01,
-1.11480080e-01,-2.71273434e-01,-3.10734630e-01,-4.22358245e-01,-2.24290296e-01,
-6.70127809e-01,9.12336353e-03,-2.63327193e-02,-3.02372966e-02,-1.18291959e-01,
-2.05386981e-01,-3.55436444e-01,-3.48081529e-01,-4.26881500e-02,-1.20400704e-01,
3.88500420e-03,5.71425036e-02,1.06352769e-01,-1.82443693e-01,8.31200406e-02,
4.75700498e-02,-3.23368609e-01,-2.96766847e-01,-8.83940905e-02,-1.62045643e-01,
-2.31500998e-01,5.35260215e-02,-1.86493397e-01,-6.37919605e-02,-1.37693971e-01,
1.82491675e-01,-1.78755969e-01,-5.51502347e-01,-6.17329955e-01,-6.71198815e-02,
5.62004803e-04,2.02373222e-01,-7.03533471e-01,-1.77087098e-01,-1.49850205e-01,
-2.64885247e-01,-2.44953185e-01,-5.95568530e-02,-1.92506939e-01,-9.97303352e-02,
2.64787495e-01,-7.17195123e-02,-6.83852881e-02,2.60430992e-01,1.95701588e-02,
1.02887027e-01,1.09624332e-02,-2.79758964e-02,3.51559394e-03,-1.71317980e-01,
-1.84832349e-01,-9.78355855e-02,1.73128858e-01,2.03836247e-01,-9.37647652e-04,
9.70135480e-02,1.20644234e-01,1.24626949e-01,5.05394600e-02,1.46580988e-03,
5.69401942e-02,9.32412297e-02,2.00073153e-01,4.52636182e-02,1.54284611e-02,
-3.31956834e-01,2.08289906e-01,1.22120298e-01,9.46301743e-02,3.82853970e-02,
5.49474992e-02,-7.88283646e-02,1.38621837e-01,1.09537572e-01,-7.00006604e-01,
3.74805853e-02,2.49817923e-01,-1.70366049e-01,-3.09286743e-01,-2.20218807e-01,
1.64271459e-01,-3.75150681e-01,-2.75149256e-01,2.17074007e-01,-2.63804108e-01,
-8.55396912e-02,-4.13068861e-01,-1.78379491e-02,4.85422947e-02,-1.16977610e-01,
-3.45192105e-01,1.55489728e-01,-1.54918358e-01,-4.70753275e-02,-1.33363411e-01,
-1.05220333e-01,-3.97923619e-01,-4.38102543e-01,2.31181368e-01,1.71328098e-01,
1.55556630e-02,-7.24056244e-01,-4.23493356e-01,1.39570177e-01,-1.67091668e-01,
-4.79961820e-02,1.19447850e-01,9.53099206e-02,7.83765689e-02,-1.40956998e-01,
-1.73485111e-02,-2.32217699e-01,-4.83732373e-02,1.39626081e-03,-8.27905163e-02,
-1.32191509e-01,2.54045874e-01,-4.62252676e-01,-2.11441115e-01,-3.15369666e-01,
-4.84530888e-02,-1.54772803e-01,-5.09779334e-01,-2.67000467e-01,-2.40413612e-03,
-1.44334108e-01,-6.06187463e-01,-1.84601828e-01,-4.04475749e-01,-3.23301479e-02,
-3.10212672e-01,2.80636072e-01,-2.52133995e-01,-1.33377254e-01,-2.49838918e-01,
-3.84296566e-01,-4.14262107e-03,-9.60721727e-03,-4.18497145e-01,-8.23950469e-02,
-1.66058242e-02,1.22860245e-01,3.16701233e-01,-8.21326002e-02,1.59601018e-01,
-3.24303240e-01,1.07123507e-02,-4.14205976e-02,-2.09006250e-01,-5.07799834e-02,
4.56861444e-02,-1.14452150e-02,7.81823024e-02,8.55946988e-02,4.51265723e-02,
1.98269829e-01,-2.09929839e-01,7.92921484e-02,4.94367108e-02,-3.50306123e-01,
-1.37988970e-01,7.40801692e-02,-2.58199573e-01,-1.78439289e-01,-2.75313586e-01,
-2.30300292e-01,-7.84130394e-03,-4.30980027e-01,1.70068726e-01,-1.43985048e-01,
2.57794678e-01,9.40826982e-02,-1.99616894e-01,-1.29678501e-02,1.69058949e-01,
2.92210072e-01,-1.85291365e-01,3.22985440e-01,8.76634941e-02,1.10831819e-01,
3.15382898e-01,1.23090453e-01,9.26606953e-02,4.83863540e-02,2.16649279e-01,
2.66312033e-01,-8.21135342e-02,2.03588441e-01,4.01171744e-01,3.88872534e-01,
4.28252704e-02,4.96672541e-02,2.28197575e-01,1.04984343e-01,3.08606267e-01,
2.96284050e-01,-5.97009435e-02,1.24589756e-01,1.52405575e-01,3.21623385e-01,
6.90909922e-01,3.86857331e-01,7.07710488e-03,-3.03625800e-02,6.03703082e-01,
3.28176767e-01,-1.86299086e-01,2.75081079e-02,5.33438884e-02,-1.16812162e-01,
-1.10259719e-01,-2.65322268e-01,-1.59959778e-01,-1.75303653e-01,-4.23174441e-01,
-2.49138996e-01,-1.87870502e-01,-2.26476535e-01,-2.53742840e-02,5.70279919e-02,
6.19933195e-02,-4.50918883e-01,-2.00288028e-01,7.60789365e-02,1.63089365e-01,
-2.71460086e-01,-1.34691939e-01,-3.28867912e-01,-2.75404215e-01,-5.19228280e-02,
-2.37882927e-01,-5.98347224e-02,1.91811949e-01,-2.96119571e-01,-3.98457170e-01,
-1.48803413e-01,5.08913025e-02,-4.21217948e-01,1.06108338e-02,-2.76440769e-01,
-2.70022023e-02,9.75535586e-02,1.30999252e-01,4.03954983e-02,-1.31722942e-01,
1.39410824e-01,-7.21354038e-02,-2.08355367e-01,-1.69066831e-01,1.28614977e-01,
-1.90285593e-01,-2.63234824e-01,1.41870812e-01,7.08246008e-02,4.58969384e-01,
9.71214175e-02,2.41442602e-02,3.57378364e-01,-4.20696527e-01,-2.12692082e-01,
1.56569883e-01,1.99436963e-01,3.13760601e-02,-5.34404874e-01,1.77102853e-02,
-3.23538274e-01,-1.30914059e-02,-1.17299564e-01,-7.08652660e-04,-2.75870144e-01,
-1.10646963e-01,2.40623862e-01,-7.24553987e-02,-1.59369752e-01,-9.78133902e-02,
1.06004715e-01,-1.59548774e-01,-1.92860201e-01,4.14753109e-02,-3.68366912e-02,
-3.70193310e-02,1.59981772e-01,-9.47704762e-02,1.40783447e-03,-1.39497310e-01,
-2.29497403e-02,-1.50087193e-01,3.97055864e-01,4.17264193e-01,2.97015131e-01,
2.07243189e-01,-1.88044533e-01,-1.90201372e-01,-6.32462740e-01,-5.49582779e-01,
-3.15445289e-02,4.72771078e-02,4.71221544e-02,7.89613649e-02,2.89963335e-02,
-2.77068287e-01,1.10692546e-01,3.24730039e-01,-1.13163933e-01,2.90554278e-02,
-4.49255466e-01,-3.07140294e-02,-3.88437539e-01,-1.89049259e-01,2.01582350e-02,
-2.95202434e-01,3.60347122e-01,-7.35538229e-02,4.95047905e-02,9.72318798e-02,
5.60188554e-02,1.31171122e-02,-3.75271291e-01,-1.18064687e-01,-2.95045953e-02,
3.32144260e-01,3.86965841e-01,-5.37882268e-01,-7.18368813e-02,-9.61493049e-03,
-1.06198974e-01,-1.92749992e-01,-4.34185833e-01,-1.46783084e-01,-1.60735443e-01,
9.49940026e-01,-2.38232553e-01,2.42736369e-01,1.77380949e-01,-2.66632676e-01,
-2.59443037e-02,6.72459081e-02,-2.01519161e-01,9.72278267e-02,1.23090081e-01,
-3.16116601e-01,-1.51528060e-01,-4.60455000e-01,-3.80544811e-02,-9.24956426e-02,
1.13259636e-01,1.43828765e-01,2.97399253e-01,7.81532228e-02,-2.15858191e-01,
-3.71876694e-02,2.52740502e-01,-3.97440374e-01,1.06868550e-01,3.02199423e-02,
-2.49866098e-02,1.91194549e-01,-3.59389275e-01,-4.52894956e-01,-3.41918170e-01,
1.33099422e-01,-8.44626799e-02,-9.59061384e-02,-1.74831077e-01,-5.21578014e-01,
-5.20802677e-01,-1.13239609e-01,-1.68187097e-01,-2.22269624e-01,-4.22045849e-02,
-2.74791688e-01,1.98332533e-01,3.48715903e-03,-1.76277041e-01,-4.11560610e-02,
-3.37950975e-01,-3.20770591e-03,-1.72723457e-02,2.25635186e-01,-3.13583255e-01,
1.06856659e-01,3.97801250e-01,-1.37690812e-01,-2.29979176e-02,-2.04186305e-01,
-7.42014199e-02,5.06184511e-02,-1.69884697e-01,-2.27026805e-01,4.38959986e-01,
3.24733136e-03,2.91261692e-02,-4.64995712e-01,-1.07699543e-01,-4.56750542e-02,
3.45322609e-01,1.82099357e-01,1.03327245e-01,-2.57345527e-01,-6.98452145e-02,
6.27839565e-02,-7.68762827e-02,1.66468814e-01,-5.22529818e-02,6.99945986e-02,
3.36650535e-02,4.21950221e-01,3.43416929e-02,-1.55554742e-01,-6.16457611e-02,
1.27405763e-01,2.85874665e-01,-3.97588722e-02,-4.59573343e-02,2.63379663e-01,
3.04440886e-01,3.34243476e-01,1.23588676e-02,6.93810165e-01,2.81473488e-01,
2.12766349e-01,-8.71510804e-02,9.13209692e-02,4.28234726e-01,2.76352704e-01,
6.95468247e-01,3.52720320e-01,7.17689991e-02,-3.08640748e-01,3.77461135e-01,
3.81189585e-01,2.45748252e-01,6.01755083e-01,-1.19616957e-02,2.00673625e-01,
4.29494888e-01,7.19196536e-03,-2.31617153e-01,8.98301974e-02,9.97518674e-02,
1.14134839e-02,-5.22924103e-02,4.25552949e-02,1.45935059e-01,-1.30470470e-01,
-8.44508782e-02,-6.11384027e-02,4.31183577e-02,3.80108654e-01,-2.27898620e-02,
1.43091857e-01,3.19503605e-01,2.57759124e-01,-1.75413683e-01,1.66843116e-01,
-2.15308778e-02,-7.19519183e-02,2.16022581e-01,3.93954366e-01,-2.23366588e-01,
7.73406699e-02,1.69523507e-02,-2.53212720e-01,1.16975605e-01,-1.95969597e-01,
-2.04702094e-01,-1.24645576e-01,1.27222314e-01,-1.41293660e-01,2.48741642e-01,
3.97639461e-02,1.89542279e-01,2.09339648e-01,-3.99431884e-02,2.66892642e-01,
3.68844390e-01,8.29487368e-02,2.50839978e-01,1.07000500e-01,2.35532016e-01,
1.15687057e-01,1.76203892e-01,2.78813895e-02,3.33975852e-02,6.89419806e-01,
-1.92386344e-01,3.06290239e-02,4.27452147e-01,1.13771796e-01,-1.02206189e-02,
6.58900976e-01,2.53783073e-02,1.27898932e-01,-4.78024688e-03,1.92737445e-01,
-9.93562043e-02,-7.33866822e-03,1.74967676e-01,3.60966623e-01,2.00267419e-01,
1.54073611e-01,7.66133666e-02,2.55686998e-01,-3.73270959e-01,-6.46017641e-02,
1.05122462e-01,-8.85050744e-02,1.72006443e-01,-1.54524505e-01,1.02727130e-01,
2.95114487e-01,2.26966709e-01,2.65086621e-01,-1.85223371e-01,1.15354639e-02,
4.39649224e-01,4.25094754e-01,-9.82372761e-02,8.60024523e-03,1.44076765e-01,
2.57139325e-01,3.89168769e-01,8.74309614e-02,1.90485530e-02,1.48348017e-02,
8.83904815e-01,2.70205408e-01,8.11767653e-02,5.66884935e-01,2.30982780e-01,
-3.73096317e-02,3.26965839e-01,-6.12418056e-02,-4.73562218e-02,1.68720379e-01,
3.11314255e-01,-2.37497121e-01,-5.73279969e-02,2.71295831e-02,-8.16679224e-02,
6.33544773e-02,1.46337450e-01,3.73069733e-01,-8.34443718e-02,-4.26199362e-02,
1.93040177e-01,1.23175293e-01,2.46290620e-02,2.48475805e-01,-8.17478076e-02,
1.90940857e-01,-9.42066014e-02,2.55100757e-01,1.48157686e-01,2.63055593e-01,
2.79362500e-01,6.04978383e-01,1.12253480e-01,1.13427296e-01,3.45527567e-02,
-3.33694220e-01,1.57887623e-01,1.07457682e-01,-1.76530406e-01,-4.03417787e-03,
-3.92823182e-02,1.60230935e-01,1.28679499e-01,7.08619580e-02,1.74077705e-01,
-1.57893106e-01,-5.68118617e-02,7.37307295e-02,1.03965163e-01,4.18608695e-01,
-1.05427437e-01,4.11860675e-01,2.41334930e-01,2.31874928e-01,4.47274186e-02,
2.06042752e-02,2.17312902e-01,1.08544543e-01,3.02564621e-01,1.09230474e-01,
2.76657850e-01,1.60282686e-01,3.77581626e-01,2.94478655e-01,-1.11645535e-02,
7.11985081e-02,1.43111095e-01,-2.51981258e-01,2.21450821e-01,1.08552985e-01,
8.91814455e-02,6.16553798e-02,-5.09033129e-02,2.34252006e-01,-9.27024428e-03,
5.37571982e-02,-1.79542452e-01,2.55269825e-01,9.33510587e-02,2.11513281e-01,
2.01214068e-02,2.01619372e-01,-2.70325392e-01,-9.64257494e-02,6.65557832e-02,
2.16423720e-01,1.98815793e-01,3.20744008e-01,3.98893580e-02,4.13344115e-01,
1.92591816e-01,2.28534430e-01,2.92043835e-01,-2.94317566e-02,4.02431577e-01,
1.06021062e-01,9.32306498e-02,-8.58337805e-02,3.95911783e-02,-1.82630286e-01,
2.43568242e-01,-4.80167009e-02,1.58636436e-01,1.66975811e-01,-2.67325848e-01,
5.42207837e-01,4.88145091e-02,1.36174917e-01,6.76617622e-02,6.60102954e-03,
-2.23728027e-02,-5.27933724e-02,-2.53440347e-02,3.49712409e-02,-3.29491496e-02,
1.36652559e-01,-7.91317150e-02,9.24533308e-02,-1.23007514e-01,-1.16176538e-01,
-3.72794718e-02,6.67132214e-02,-1.08463459e-01,-1.80175513e-01,1.23084700e-02,
1.12109818e-01,1.88702159e-02,1.36396632e-01,2.98945576e-01,1.22730754e-01,
1.59357101e-01,-2.62490138e-02,-3.26723084e-02,3.64837125e-02,1.46168604e-01,
-2.12034807e-01,-3.53302360e-02,-3.06532364e-02,4.52498794e-02,2.52031416e-01,
9.46766604e-03,4.10984084e-02,5.65476269e-02,1.66700035e-01,-8.52275938e-02,
2.26751715e-02,-4.12024111e-02,5.81700914e-02,6.76563848e-03,7.01569244e-02,
-1.59903944e-01,-4.78765555e-02,-7.33492300e-02,1.30971104e-01,1.30953208e-01,
-4.21045013e-02,-2.26544566e-04,1.67246744e-01,8.55141971e-03,1.06228575e-01,
4.10685390e-02,-7.11730048e-02,-1.75147414e-01,-1.37098670e-01,2.23228097e-01,
-1.59239411e-01,-1.40403315e-01,1.15955457e-01,6.71751797e-02,-3.40567566e-02,
3.20100300e-02,1.47674428e-02,3.37357335e-02,-1.47018969e-01,-2.46145070e-01,
-1.22638777e-01,-2.28113568e-04,-6.99468032e-02,-3.45008969e-02,1.12859845e-01,
-7.89990574e-02,-1.31055951e-01,-8.45949445e-03,-3.73115689e-02,5.42839952e-02,
8.93023312e-02,2.19525807e-02,6.87287971e-02,2.32580844e-02,1.62989780e-01,
2.92858277e-02,5.57773164e-04,-6.03087693e-02,6.98094517e-02,-2.36957911e-02,
8.64088442e-03,-4.58538719e-02,1.31725252e-01,-7.33547732e-02,1.52051345e-01,
7.90852159e-02,6.64029494e-02,-1.88845843e-01,-9.89206582e-02,4.79120202e-02,
3.75966839e-02,1.30554959e-01,7.36176297e-02,-8.97889491e-04,3.22765678e-01,
-2.07871571e-01,1.42991804e-02,1.19969994e-01,2.13222764e-02,1.62356086e-02,
-1.73080593e-01,-1.34727396e-02,4.70442250e-02,-3.61450642e-01,-1.07336372e-01,
-2.28706673e-01,-1.06663153e-01,-2.19753310e-01,3.17975245e-02,2.72378679e-02,
-8.17409456e-02,-7.68377185e-02,-4.70391959e-02,-1.21133260e-01,-1.50166929e-01,
2.04958506e-02,-1.78306460e-01,-5.36308810e-02,-6.61522076e-02,5.15879914e-02,
6.48789480e-02,3.81479785e-02,2.01776877e-01,-1.67395204e-01,-1.92909986e-01,
-1.83220848e-01,-4.89262156e-02,3.09768409e-01,-5.84339797e-02,-4.22663316e-02,
4.91353907e-02,2.18558118e-01,-2.13485956e-02,-7.00032786e-02,1.44739762e-01,
1.49110824e-01,6.46527633e-02,5.12610227e-02,1.30832002e-01,-3.51991830e-03,
1.06607769e-02,1.50839776e-01,3.19033325e-01,-3.85126472e-02,1.86574429e-01,
-9.46918204e-02,2.95701802e-01,-1.24442220e-01,1.16093315e-01,-1.12936497e-01,
1.73745379e-01,-1.47849515e-01,3.72455627e-01,-5.87330908e-02,-2.43255362e-01,
4.00737345e-01,6.57374188e-02,7.13987201e-02,3.99622433e-02,4.10662591e-02,
-3.48918550e-02,-8.14798288e-03,4.11000587e-02,-5.48828917e-04,2.31511712e-01,
-2.77140178e-02,2.77470667e-02,-1.10199340e-01,1.13021336e-01,1.84895918e-01,
2.45227724e-01,4.00694728e-01,1.29006580e-01,-1.55681506e-01,-1.64980330e-02,
2.64394376e-02,-6.79945126e-02,2.34450638e-01,3.71059835e-01,5.08072637e-02,
1.18987337e-01,1.07013360e-01,1.86717629e-01,3.07568107e-02,1.96075067e-01,
3.38028073e-01,3.12294453e-01,1.38015345e-01,-1.00806884e-01,-2.37067584e-02,
3.64746861e-02,-1.36236593e-01,3.92527938e-01,1.49845228e-01,7.82450363e-02,
3.19091529e-01,2.05298632e-01,-2.71420088e-02,-7.16998354e-02,-6.30735531e-02,
1.30539119e-01,6.61400408e-02,1.52889639e-01,3.08972299e-01,1.42493188e-01,
3.11131388e-01,9.77588594e-02,1.07539855e-01,1.53165489e-01,1.96632653e-01,
-4.84993272e-02,2.38094494e-01,3.89680386e-01,-6.83694109e-02,4.97342438e-01,
2.07568318e-01,-2.00960296e-03,-6.63184226e-02,4.91980195e-01,1.81262940e-01,
1.59246758e-01,4.51134771e-01,1.41834006e-01,-4.70865667e-02,2.05960512e-01,
-1.15840279e-01,2.51967490e-01,-2.84540020e-02,1.62614241e-01,2.58004665e-02,
4.87399548e-02,1.01481803e-01,-2.01051548e-01,5.16689599e-01,1.79258317e-01,
-1.52001694e-01,-2.50743568e-01,3.79534155e-01,-2.46146079e-02,-3.10726017e-01,
2.31335387e-01,-7.39789084e-02,-1.15160219e-01,-1.33162960e-01,-4.19625193e-01,
6.15111999e-02,8.63009617e-02,-2.56810803e-03,-9.66586843e-02,1.19051583e-01,
1.82626918e-02,-5.98281547e-02,1.17930762e-01,1.18477784e-01,8.88547376e-02,
-1.26956463e-01,-1.30173549e-01,3.04144174e-01,2.62973487e-01,3.75704855e-01,
-1.91363692e-02,1.61164895e-01,6.11548424e-02,2.05579892e-01,-1.64042383e-01,
-4.06770647e-01,8.09419453e-02,4.01539803e-01,-6.80262446e-02,1.13306411e-01,
-3.79059054e-02,-6.87103197e-02,-4.82226126e-02,-1.26626581e-01,5.91764636e-02,
1.23499162e-01,-5.18003060e-03,-9.44118723e-02,3.01252544e-01,-6.51405156e-02,
1.37512684e-01,-1.71879262e-01,-4.57833558e-02,-1.28703751e-03,-1.50452906e-04,
1.67979300e-02,-4.29114141e-02,1.46114856e-01,1.36488853e-02,-1.01938717e-01,
-2.34396923e-02,-1.47324326e-02,-6.77699149e-02,-8.23217854e-02,2.68440455e-01,
-8.62850845e-02,1.36181742e-01,-2.09289089e-01,1.13033824e-01,-9.41583514e-03,
1.15276083e-01,-1.05872244e-01,1.21211587e-03,1.17766790e-01,6.54367059e-02,
-1.58637673e-01,-1.72495302e-02,9.58076492e-03,-1.77582055e-01,-1.69508867e-02,
3.22746873e-01,7.32540712e-02,-8.72133225e-02,2.75695562e-01,5.03754504e-02,
3.95106040e-02,2.99268305e-01,2.79888988e-01,4.37153727e-01,1.58628464e-01,
-1.80248134e-02,2.74441063e-01,-9.24171433e-02,-2.73242202e-02,1.24819860e-01,
5.21710455e-01,-9.90834087e-02,-9.27990973e-02,4.30625528e-02,1.07919492e-01,
-9.52844620e-02,5.88492639e-02,2.21907377e-01,-1.75865129e-01,-4.69244197e-02,
3.63792554e-02,-2.04632238e-01,-8.56235996e-02,-1.24282487e-01,-1.06124371e-01,
-8.87687355e-02,-7.88683519e-02,-2.61204150e-02,-1.45278573e-01,-1.85338482e-02,
3.46159935e-02,1.32706642e-01,-1.19185671e-01,-3.36114061e-03,-1.05040446e-01,
3.34365629e-02,-9.50038284e-02,2.61673659e-01,1.47576809e-01,1.82026803e-01,
-9.87434387e-02,6.69558272e-02,4.67739888e-02,1.08298689e-01,1.13677382e-02,
1.27677023e-01,1.92356165e-02,3.45679186e-02,-1.34486765e-01,-2.50602178e-02,
8.84089172e-02,1.13780171e-01,-4.08657826e-02,-3.10919639e-02,2.00210258e-01,
4.56460118e-02,1.41739130e-01,2.24723727e-01,-3.01215984e-02,4.46938090e-02,
1.84108987e-02,1.88108996e-01,-2.58839596e-03,1.16515858e-03,2.27258474e-01,
1.59667417e-01,7.27724377e-03,1.50691554e-01,1.02553666e-01,2.65268624e-01,
-2.98816375e-02,1.83989778e-01,1.03629045e-01,1.63368523e-01,-1.51455551e-01,
2.82957345e-01,-8.30112025e-02,1.34641707e-01,1.51172830e-02,-1.94233228e-02,
1.86768204e-01,2.04705615e-02,-1.41864970e-01,7.26107135e-03,8.67078954e-04,
3.49218734e-02,-5.01896814e-02,4.76730913e-02,-9.40234587e-02,-1.28464088e-01,
1.66337402e-03,-4.06856723e-02,-1.36910781e-01,9.98869240e-02,4.41973284e-02,
1.74627125e-01,-4.53391448e-02,1.84874207e-01,4.48789187e-02,1.98590964e-01,
3.49648714e-01,3.55211020e-01,3.19804363e-02,-1.87885053e-02,-2.68454198e-02,
4.83254306e-02,-1.81275140e-02,-1.10338300e-01,1.69403866e-01,-2.06981584e-01,
-8.61741975e-02,2.15855502e-02,3.49798277e-02,-6.62962068e-03,-1.32004604e-01,
1.79951683e-01,-3.17864865e-01,-6.46347404e-02,-3.12612772e-01,2.60926455e-01,
-1.57796502e-01,-1.04125291e-01,4.09351103e-02,-1.90410558e-02,6.45751879e-02,
-4.47008424e-02,9.18757096e-02,6.81152940e-02,-1.44461393e-01,-1.88423291e-01,
2.95700729e-01,-2.76250392e-01,-1.28372788e-01,-1.84414834e-01,-4.42994714e-01,
-5.55364089e-03,3.69873755e-02,-1.24424674e-01,-1.34741142e-01,-2.13314995e-01,
-1.35521382e-01,-3.62114459e-02,-1.27267271e-01,-1.52502537e-01,9.08206701e-02,
4.11857329e-02,2.59294659e-01,-1.44341826e-01,-2.30668802e-02,-1.32232890e-01,
-1.11694217e-01,-3.83862033e-02,-1.33751193e-02,-5.61534576e-02,-1.10647231e-01,
-7.79823959e-02,-1.17198430e-01,6.07782938e-02,3.54845189e-02,7.58280605e-02,
-1.86205670e-01,-1.83291897e-01,-3.32020670e-01,7.13146031e-02,1.22396968e-01,
-5.95298335e-02,1.05809033e-01,3.63627523e-01,1.22561477e-01,-2.96397787e-02,
1.37563989e-01,-5.73520064e-02,-1.10332154e-01,-6.72767609e-02,8.70498121e-02,
5.79322912e-02,1.59166873e-01,1.03568137e-02,1.94768772e-01,3.01252782e-01,
7.42185628e-03,-1.20638937e-01,4.05522585e-01,1.33233905e-01,-3.29250187e-01,
-1.21496513e-01,2.11251423e-01,1.35603055e-01,-4.75429446e-02,2.65806049e-01,
1.39851928e-01,1.61451846e-02,-1.35008544e-01,2.74001598e-01,-3.33339088e-02,
-5.42649627e-02,2.35929072e-01,1.67854756e-01,-1.53541133e-01,1.36106357e-01,
3.61160636e-01,3.87990698e-02,1.46895528e-01,1.13774814e-01,-4.38096561e-02,
-4.83653769e-02,5.97519167e-02,2.28449225e-01,-3.07526678e-01,-1.18523315e-01,
-7.47911483e-02,-3.50959376e-02,7.98046291e-02,4.54812385e-02,3.51329311e-03,
4.27083187e-02,-1.72114260e-02,5.68198204e-01,-8.81480798e-02,1.61918014e-01,
-2.04449177e-01,1.14333451e-01,1.34823518e-02,-1.86460018e-01,1.37741461e-01,
-4.23217267e-02,-2.04303429e-01,7.18681440e-02,-5.61563745e-02,1.24689229e-01,
5.30221872e-02,-1.75828472e-01,2.69071251e-01,-6.56914786e-02,5.64153492e-03,
1.76449530e-02,-5.12648150e-02,2.16564626e-01,4.09651190e-01,7.15346634e-02,
1.86139807e-01,6.13518283e-02,-1.16935094e-04,-2.38205910e-01,-4.03854549e-02,
-5.80810048e-02,-1.45659253e-01,9.28146318e-02,-6.97590187e-02,1.35792166e-01,
1.77442744e-01,-5.95573857e-02,-2.82178614e-02,-1.12830304e-01,-1.25716135e-01,
-3.13079730e-03,1.03882298e-01,3.41118872e-02,-1.83299363e-01,-1.33471936e-01,
-4.47071083e-02,-6.10012263e-02,4.81269956e-01,-2.16248497e-01,-1.32032663e-01,
1.98886290e-01,4.73300591e-02,-1.08961083e-01,3.02051067e-01,1.98034346e-01,
8.15457329e-02,4.30603623e-02,-3.30917537e-01,-1.40009418e-01,-3.72777134e-02,
-2.54783742e-02,2.51149833e-01,3.71052064e-02,-4.04677317e-02,-2.66988844e-01,
8.03991854e-02,-1.06703050e-01,-4.94092517e-02,4.59672650e-03,-2.80471027e-01,
-3.47773731e-02,-4.41229753e-02,1.30232824e-02,2.59381086e-01,7.87682831e-02,
-2.76920140e-01,1.62255543e-03,-5.11825569e-02,-1.14317454e-01,4.00529727e-02,
-1.07570745e-01,1.80907294e-01,2.38702014e-01,1.49253070e-01,1.76722541e-01,
2.16900051e-01,-1.18183240e-01,2.27726042e-01,1.89430322e-02,-1.13796115e-01,
-2.70276099e-01,1.71691358e-01,1.32801414e-01,1.56269312e-01,-8.89932439e-02,
-1.18147194e-01,2.01960608e-01,5.72903715e-02,2.11736977e-01,7.18594808e-03,
3.29685882e-02,9.26964208e-02,-3.44161652e-02,-9.04835388e-02,-5.76750301e-02,
-8.33771750e-02,2.43431516e-02,3.14219445e-02,-3.46364919e-03,1.86755601e-02,
-2.27842644e-01,-5.77677786e-03,1.11653604e-01,9.18036327e-02,-1.47175685e-01,
-2.06894636e-01,-5.92705831e-02,-3.40759844e-01,-1.69780999e-01,2.22945705e-01,
-1.82738259e-01,2.81360745e-01,3.10658067e-02,4.08099890e-01,1.01791054e-01,
-6.71671182e-02,7.47129647e-03,1.81535989e-01,-3.75440158e-02,1.99525476e-01,
-3.62039596e-01,-1.36494711e-01,-1.45507023e-01,1.03283115e-01,-1.67276517e-01,
-3.43862772e-02,6.71739653e-02,2.57608652e-01,-1.85226813e-01,1.82040811e-01,
5.88171147e-02,3.42035154e-03,4.70548533e-02,2.50337154e-01,-1.34984031e-01,
-1.33466110e-01,-2.34809835e-02,1.20525904e-01,1.99424535e-01,2.22766712e-01,
2.35203668e-01,4.17349398e-01,-1.22989088e-01,-2.07046211e-01,1.11559015e-02,
1.99080572e-01,-1.63189620e-01,3.79419327e-01,3.95647176e-02,1.07303545e-01,
-1.16827693e-02,-1.35709837e-01,2.27375358e-01,-3.19063477e-02,-2.52203289e-02,
1.21259265e-01,8.96535255e-03,8.66123065e-02,2.83845574e-01,-4.37472053e-02,
-8.56586844e-02,1.14587173e-01,-1.21519282e-01,2.66806066e-01,1.37073562e-01,
8.81608874e-02,-4.59802225e-02,1.36783481e-01,-2.92820126e-01,8.46697837e-02,
6.40407428e-02,1.23993717e-01,1.42409131e-01,2.90922672e-01,4.31005508e-01,
6.25144169e-02,1.92961857e-01,-6.93978369e-02,6.99850842e-02,1.88866660e-01,
1.12851709e-01,-6.74279556e-02,1.46379322e-01,1.51635140e-01,-1.05076842e-01,
1.81770921e-01,-6.68485835e-02,9.02897194e-02,2.34731901e-02,2.65221447e-01,
2.61547953e-01,7.15691894e-02,1.14728272e-01,5.70538118e-02,-1.79704830e-01,
1.87105820e-01,-3.03277187e-02,2.52818435e-01,1.04469426e-01,2.39099532e-01,
-1.17366970e-01,-8.63463655e-02,-3.54502976e-01,-5.86218946e-02,-3.92432123e-01,
2.90674418e-01,4.50840443e-01,3.51551831e-01,4.00562257e-01,2.69182157e-02,
1.55714517e-02,-1.93321303e-01,-2.16112942e-01,8.14753398e-02,-3.21656227e-01,
2.75965743e-02,-1.42399907e-01,-4.59457375e-02,-1.16618201e-01,-8.05979893e-02,
-1.26312096e-02,1.21322855e-01,1.07487701e-01,-2.53190219e-01,6.52370527e-02,
-1.34827882e-01,-2.10652053e-01,-2.65187085e-01,-1.99264705e-01,1.88341543e-01,
3.52248341e-01,-6.13859445e-02,-1.20254561e-01,-1.74317703e-01,-1.89197630e-01,
-8.98711830e-02,1.75909415e-01,-1.12234101e-01,-2.60658383e-01,-1.90286398e-01,
-9.76983607e-02,1.60726998e-02,-2.94439882e-01,-2.15124995e-01,-1.22525148e-01,
8.70887190e-02,2.23951519e-01,1.70288365e-02,1.18213343e-02,-6.97118267e-02,
1.58005372e-01,-8.81363347e-04,-7.90271312e-02,5.82993366e-02,2.31096193e-01,
6.13037229e-01,4.83885527e-01,3.33603770e-01,5.96453436e-02,-1.91940352e-01,
-7.15182945e-02,5.42005803e-03,1.47328317e-01,1.35066837e-01,-1.33844644e-01,
1.89124450e-01,1.66277975e-01,-4.09343034e-01,5.04849255e-02,-9.28004310e-02,
-9.08618271e-02,-2.36292720e-01,4.65403765e-01,3.77533346e-01,1.24302000e-01,
1.85404286e-01,-3.31824943e-02,-3.45027298e-01,-2.68084824e-01,-2.84404814e-01,
2.48607136e-02,-7.94761851e-02,-2.95909613e-01,1.29183069e-01,-1.58796147e-01,
-1.34847715e-01,7.02911913e-02,-1.39556929e-01,6.71871006e-03,1.09130889e-01,
1.09741844e-01,-2.68013090e-01,-2.75066584e-01,-2.23170891e-01,-3.12124379e-02,
1.54793724e-01,-3.87227163e-02,1.93648767e-02,2.23988593e-01,1.19473496e-02,
-1.73559830e-01,8.45782608e-02,-1.73872814e-01,1.26727000e-01,-1.20067686e-01,
2.00451300e-01,1.04709854e-02,3.31312090e-01,4.85192597e-01,-1.34746253e-01,
-2.94706583e-01,-1.62143797e-01,-3.49791139e-01,1.89118296e-01,-1.03358980e-02,
2.99553752e-01,1.05611943e-02,1.44252375e-01,8.89350995e-02,-8.49447101e-02,
3.43639195e-01,-9.24263671e-02,1.22013919e-01,2.62300104e-01,2.84755826e-01,
1.54290870e-01,-2.70458341e-01,3.74803215e-01,-1.56622287e-02,1.22136422e-01,
-1.98752154e-02,2.59891897e-01,-8.35684985e-02,1.79526895e-01,4.06256970e-03,
3.30255747e-01,-1.48503333e-01,-1.65703148e-01,3.50765921e-02,1.03985526e-01,
1.98440790e-01,4.00953203e-01,1.06058913e-04,-5.18283471e-02,6.97399855e-01,
6.36658192e-01,-9.53528881e-02,8.08189437e-02,3.60797673e-01,-2.44102985e-01,
5.82252443e-01,1.32283881e-01,-3.04063875e-03,3.03322107e-01,2.77392060e-01,
1.08138904e-01,-4.20560054e-02,1.45266756e-01,3.75200361e-01,1.66481078e-01,
1.22156292e-01,4.45014209e-01,1.37038216e-01,1.44559234e-01,2.09994972e-01,
1.27061471e-01,-2.21281573e-01,1.25788167e-01,1.88575402e-01,3.09491873e-01,
8.15847039e-01,6.24491274e-01,6.38001680e-01,1.18654720e-01,1.52747452e-01,
-3.12697217e-02,1.06272526e-01,1.25833869e-01,4.60940927e-01,-3.24832946e-02,
-1.76345762e-02,-1.94550887e-01,-8.35394710e-02,2.48784691e-01,3.81838113e-01,
-1.99050289e-02,2.32063636e-01,3.89433205e-02,1.93034440e-01,1.25475526e-01,
2.45296150e-01,-1.31703028e-02,3.82006876e-02,1.72317386e-01,6.54851794e-02,
4.22222652e-02,-8.57887343e-02,3.81338131e-03,1.88595522e-02,-2.06185542e-02,
-1.37014046e-01,4.55064401e-02,3.44129167e-02,5.83208837e-02,-1.33477569e-01,
8.24513286e-02,-2.74246167e-02,-5.74268661e-02,-2.04745904e-01,1.42401367e-01,
1.50691509e-01,-1.63928390e-01,-2.79483031e-02,1.50501067e-02,-5.83790056e-02,
5.91429994e-02,-5.61845824e-02,2.07234129e-01,1.12717606e-01,3.15907955e-01,
-7.77159110e-02,5.33997267e-02,1.88093022e-01,1.53917763e-02,-1.44623414e-01,
-3.66808847e-02,2.64119208e-01,-1.28947245e-02,3.15919995e-01,-6.53878152e-02,
2.20091529e-02,5.79175465e-02,3.02502096e-01,2.44465038e-01,3.30041647e-01,
-2.26527840e-01,4.69111979e-01,5.62082566e-02,3.34524602e-01,2.03045122e-02,
2.61006504e-01,1.75425872e-01,1.93190381e-01,3.29579443e-01,2.14072496e-01,
1.51103109e-01,2.00167507e-01,3.10850561e-01,4.54117311e-03,1.95915967e-01,
-1.39193490e-01,5.21673262e-02,1.25196859e-01,3.22434038e-01,1.12653010e-01,
-1.92144997e-02,2.28127524e-01,1.03772737e-01,7.83789679e-02,5.02593853e-02,
2.65502155e-01,7.27046654e-03,7.46603012e-02,4.29020584e-01,-1.94447055e-01,
-3.64248385e-03,1.98086202e-01,1.95342541e-01,3.19540918e-01,3.74934316e-01,
8.64669308e-02,3.27891022e-01,-1.51651740e-01,8.16638172e-02,-3.96345891e-02,
6.39354810e-02,-5.37231378e-02,4.23220009e-01,-2.31577884e-02,-1.17725963e-02,
8.45047459e-02,1.33464530e-01,1.50878504e-01,2.72759236e-02,-4.38154191e-02,
2.30387673e-01,2.44630098e-01,3.53503793e-01,-6.47821352e-02,-5.40671051e-02,
9.40359607e-02,-2.21638083e-02,1.37393847e-01,9.22505409e-02,2.11611062e-01,
-2.58800954e-01,-1.72314256e-01,-4.45864769e-03,5.70353009e-02,3.39161336e-01,
-4.07544859e-02,-2.00970173e-01,5.34156561e-01,1.37823168e-02,-1.12218849e-01,
4.10501719e-01,-9.42663476e-02,-9.39800888e-02,-4.52571884e-02,-2.29680195e-01,
-2.25714087e-01,}; 
k2c_tensor lstm_2_recurrent_kernel = {&lstm_2_recurrent_kernel_array[0],2,4356,{132, 33,  1,  1,  1}}; 
float lstm_2_bias_array[132] = {
4.90970433e-01,9.26777422e-02,3.05431217e-01,-1.25800527e-03,3.62287313e-01,
2.67028391e-01,8.73568282e-02,2.38286033e-01,2.65458375e-01,7.36389086e-02,
1.51864663e-01,2.22906262e-01,7.20618889e-02,1.16130877e-02,2.37957165e-01,
3.44097793e-01,1.33003384e-01,3.25822055e-01,8.10234323e-02,4.38299328e-01,
1.06129885e-01,2.47970432e-01,3.26602131e-01,3.03284287e-01,7.36841202e-01,
4.03718114e-01,1.79559767e-01,2.03887880e-01,6.91327155e-01,1.98941201e-01,
1.08586647e-01,2.48908177e-01,1.77928597e-01,1.49737298e+00,8.98321629e-01,
9.49596524e-01,9.98108745e-01,1.10303664e+00,1.13401818e+00,9.11730766e-01,
1.19495046e+00,1.09026706e+00,1.18290162e+00,8.76471043e-01,1.01445746e+00,
9.22558665e-01,9.95801032e-01,8.63349080e-01,8.52224171e-01,1.08281016e+00,
5.54064393e-01,7.90315866e-01,6.33958638e-01,5.89944541e-01,7.56282747e-01,
8.04661512e-01,8.66280556e-01,1.32861805e+00,8.75307202e-01,1.08383667e+00,
1.01016033e+00,5.03625035e-01,8.77465546e-01,5.70519805e-01,8.57564449e-01,
1.13760138e+00,6.51799381e-01,-3.47741872e-01,2.30001450e-01,-8.75816718e-02,
6.00598037e-01,5.02365410e-01,2.15717390e-01,4.68217522e-01,5.68718672e-01,
2.94017434e-01,2.96097070e-01,3.95970792e-01,1.28518283e-01,2.20846742e-01,
2.76470333e-01,2.97777325e-01,2.69510537e-01,-1.13872141e-01,1.81777120e-01,
5.12277722e-01,1.71825036e-01,4.30377185e-01,2.16064170e-01,2.38716722e-01,
-9.79518294e-02,1.44614264e-01,3.36771846e-01,4.22898322e-01,-2.06271671e-02,
-2.98197418e-01,-1.96832567e-01,2.63421386e-01,3.45827669e-01,4.75196213e-01,
3.15673649e-02,2.74313301e-01,-2.19069258e-03,4.18618709e-01,3.15715671e-01,
8.83097798e-02,2.76612580e-01,3.12761188e-01,6.61820620e-02,1.65494323e-01,
2.41674796e-01,-3.29483449e-02,4.05078195e-02,2.13840514e-01,3.12119395e-01,
1.07885323e-01,2.50874162e-01,2.58988552e-02,3.63771111e-01,9.59260389e-02,
2.63141692e-01,2.57825077e-01,2.78203189e-01,4.66644049e-01,3.57296646e-01,
2.81100720e-01,7.34123066e-02,6.88945830e-01,3.25403899e-01,-2.92650759e-02,
7.06224144e-02,3.26190859e-01,}; 
k2c_tensor lstm_2_bias = {&lstm_2_bias_array[0],1,132,{132,  1,  1,  1,  1}}; 

 
size_t reshape_3_newndim = 2; 
size_t reshape_3_newshp[K2C_MAX_NDIM] = {33, 2, 1, 1, 1}; 


size_t reshape_12_newndim = 2; 
size_t reshape_12_newshp[K2C_MAX_NDIM] = {33, 1, 1, 1, 1}; 


size_t reshape_13_newndim = 2; 
size_t reshape_13_newshp[K2C_MAX_NDIM] = {33, 1, 1, 1, 1}; 


size_t concatenate_4_num_tensors0 = 3; 
size_t concatenate_4_axis = -2; 
float concatenate_4_output_array[132] = {0}; 
k2c_tensor concatenate_4_output = {&concatenate_4_output_array[0],2,132,{33, 4, 1, 1, 1}}; 


size_t conv1d_1_stride = 1; 
size_t conv1d_1_dilation = 1; 
float conv1d_1_output_array[264] = {0}; 
k2c_tensor conv1d_1_output = {&conv1d_1_output_array[0],2,264,{33, 8, 1, 1, 1}}; 
float conv1d_1_padded_input_array[136] = {0}; 
k2c_tensor conv1d_1_padded_input = {&conv1d_1_padded_input_array[0],2,136,{34, 4, 1, 1, 1}}; 
size_t conv1d_1_pad[2] = {0,1}; 
float conv1d_1_fill = 0.0f; 
float conv1d_1_kernel_array[64] = {
-4.69147414e-02,-3.25587578e-04,-2.23347828e-01,-1.31947264e-01,-2.54937410e-01,
-1.20020293e-01,-1.21512767e-02,-2.06735879e-01,1.08907698e-02,-1.52633071e-03,
-1.53255329e-01,1.49345277e-02,2.08342284e-01,-9.65658054e-02,1.29993362e-02,
7.91111663e-02,9.56319422e-02,-6.14633560e-01,-2.72286292e-02,-2.28213951e-01,
-2.00681895e-01,-1.39461989e-02,6.82428107e-02,-6.63770974e-01,5.08853868e-02,
-5.53510666e-01,3.85256559e-02,-1.63347855e-01,1.80248126e-01,1.81942567e-01,
1.28031641e-01,-5.09921372e-01,-1.61160335e-01,3.56853381e-02,-5.98142087e-01,
-2.88174730e-02,-4.41758782e-02,8.17454085e-02,9.64795332e-03,-1.11358047e-01,
-5.81858568e-02,-1.86732952e-02,6.31814599e-02,-9.47916061e-02,-1.29662007e-01,
-2.62058258e-01,-1.33384809e-01,-1.03115760e-01,1.04647405e-01,-4.83848929e-01,
-6.60999179e-01,-3.80659461e-01,-4.82425779e-01,1.62790731e-01,1.25350431e-01,
-3.73479247e-01,4.39645171e-01,-7.04864621e-01,-5.19653745e-02,-7.49901012e-02,
1.86284378e-01,-1.21423930e-01,5.27414501e-01,-3.79503565e-03,}; 
k2c_tensor conv1d_1_kernel = {&conv1d_1_kernel_array[0],3,64,{2,4,8,1,1}}; 
float conv1d_1_bias_array[8] = {
-3.99479754e-02,4.94873643e-01,-2.15060279e-01,-2.06032634e-01,-3.01657468e-01,
-3.17776859e-01,-1.08924955e-01,3.35080415e-01,}; 
k2c_tensor conv1d_1_bias = {&conv1d_1_bias_array[0],1,8,{8,1,1,1,1}}; 

 
size_t conv1d_6_stride = 1; 
size_t conv1d_6_dilation = 1; 
float conv1d_6_output_array[264] = {0}; 
k2c_tensor conv1d_6_output = {&conv1d_6_output_array[0],2,264,{33, 8, 1, 1, 1}}; 
float conv1d_6_padded_input_array[136] = {0}; 
k2c_tensor conv1d_6_padded_input = {&conv1d_6_padded_input_array[0],2,136,{34, 4, 1, 1, 1}}; 
size_t conv1d_6_pad[2] = {0,1}; 
float conv1d_6_fill = 0.0f; 
float conv1d_6_kernel_array[64] = {
-1.49801904e-02,-4.21560436e-01,-3.57637048e-01,-2.62567978e-02,-4.92013320e-02,
2.01147407e-01,1.25291482e-01,-1.05606861e-01,-8.11143592e-02,1.95509627e-01,
1.40444323e-01,-2.73128673e-02,-1.79693893e-01,3.29421580e-01,2.88693279e-01,
-1.86119795e-01,3.07309210e-01,2.84546595e-02,1.98122129e-01,2.29708761e-01,
-4.65550333e-01,-7.34095126e-02,4.24065031e-02,3.37900788e-01,7.84876198e-02,
-1.01911187e+00,2.03662246e-01,2.46223465e-01,-5.14719963e-01,-8.20343256e-01,
3.45429927e-01,5.49782455e-01,-4.84301239e-01,-2.70234495e-01,-3.22542220e-01,
-1.89456955e-01,-4.00870532e-01,-1.88218921e-01,-9.57416520e-02,2.16540784e-01,
-2.35322431e-01,-8.68682489e-02,-1.53465420e-01,1.16714984e-01,-6.39330387e-01,
-1.63774401e-01,-2.30038583e-01,-1.28302857e-01,1.50368050e-01,-4.35040385e-01,
3.32582712e-01,2.81768322e-01,-1.65060505e-01,4.78145331e-02,-2.02720687e-02,
4.06196773e-01,3.44505578e-01,-7.23232150e-01,-8.10580254e-02,2.70464063e-01,
-4.51857388e-01,-8.48116338e-01,9.67085212e-02,3.45780551e-01,}; 
k2c_tensor conv1d_6_kernel = {&conv1d_6_kernel_array[0],3,64,{2,4,8,1,1}}; 
float conv1d_6_bias_array[8] = {
-4.16485280e-01,2.40218356e-01,-5.47652304e-01,-1.97464839e-01,-8.06657895e-02,
1.68744475e-01,-2.42463455e-01,-5.65927386e-01,}; 
k2c_tensor conv1d_6_bias = {&conv1d_6_bias_array[0],1,8,{8,1,1,1,1}}; 

 
size_t conv1d_2_stride = 1; 
size_t conv1d_2_dilation = 1; 
float conv1d_2_output_array[264] = {0}; 
k2c_tensor conv1d_2_output = {&conv1d_2_output_array[0],2,264,{33, 8, 1, 1, 1}}; 
float conv1d_2_padded_input_array[288] = {0}; 
k2c_tensor conv1d_2_padded_input = {&conv1d_2_padded_input_array[0],2,288,{36, 8, 1, 1, 1}}; 
size_t conv1d_2_pad[2] = {1,2}; 
float conv1d_2_fill = 0.0f; 
float conv1d_2_kernel_array[256] = {
3.56265247e-01,-3.36731941e-01,-9.75814283e-01,2.53691882e-01,1.96005851e-01,
-3.25187862e-01,5.37448823e-01,-1.58349231e-01,2.75008947e-01,5.76187730e-01,
5.61365247e-01,5.24227083e-01,-1.24690092e+00,9.88566950e-02,1.88232839e-01,
-2.78488070e-01,-5.74284554e-01,3.88053954e-02,-2.97532260e-01,-1.40703619e-01,
8.95157456e-02,1.28324658e-01,-1.37990773e-01,2.81900674e-01,7.79207796e-02,
-4.74564508e-02,1.48921907e-01,-1.71532854e-02,3.39862317e-01,-3.00604939e-01,
-1.02694400e-01,8.13029185e-02,-2.10255265e-01,-6.64419457e-02,-1.06706567e-01,
3.82561833e-02,-1.67051911e-01,-1.31282061e-01,-4.04720068e-01,-3.28313410e-02,
3.43270361e-01,-9.69645232e-02,1.46762803e-01,-1.93420559e-01,-2.19427094e-01,
-5.37282899e-02,3.68722640e-02,-1.40189439e-01,-3.82380724e-01,-1.80519179e-01,
-3.97121876e-01,8.24482366e-02,2.46921405e-01,-2.72048503e-01,-1.65986657e-01,
1.27238959e-01,-4.52955902e-01,2.21928179e-01,-7.67312288e-01,-1.88401446e-01,
-4.26741177e-03,4.67833161e-01,-2.56389916e-01,-9.37935114e-02,1.90316841e-01,
1.64862514e-01,-2.66091973e-01,2.43939921e-01,3.16273659e-01,1.56446412e-01,
3.67249608e-01,1.30175026e-02,8.75334293e-02,1.41393885e-01,-6.39531668e-03,
4.53670233e-01,-7.93335021e-01,6.07844926e-02,3.37403625e-01,-2.50112295e-01,
-1.77488253e-01,-2.04052702e-01,-6.23803675e-01,-2.92201132e-01,1.53318807e-01,
5.42232767e-02,1.84276164e-01,-2.96412021e-01,-2.77770497e-02,2.23836407e-01,
2.22145543e-01,-3.82819027e-02,1.47625223e-01,-3.24567139e-01,-1.98931888e-01,
-1.64912701e-01,-3.25960636e-01,-1.62076578e-02,-2.93252528e-01,-5.20018749e-02,
-2.24916577e-01,-1.68039203e-01,-4.81342077e-01,-1.03239894e-01,-8.66541639e-02,
-1.24148858e+00,-6.45824909e-01,9.24100578e-02,-5.39955944e-02,1.99650615e-01,
-1.31870657e-01,4.78218384e-02,-4.58102345e-01,-1.83830962e-01,1.41868312e-02,
1.10480525e-01,1.24469422e-01,4.32148352e-02,-1.74877569e-01,1.59973815e-01,
-3.54985371e-02,-4.09444571e-02,-2.13249087e-01,-4.67555940e-01,-7.78701365e-01,
1.70705810e-01,5.86881518e-01,4.63308804e-02,2.64464766e-01,5.18080257e-02,
3.53893518e-01,-1.25719933e-02,1.50945291e-01,1.92286775e-01,-2.68553436e-01,
3.40895861e-01,3.07459801e-01,-1.83482662e-01,3.13324444e-02,5.41277051e-01,
-3.40536445e-01,1.63270622e-01,3.07450354e-01,8.73592719e-02,3.18216570e-02,
4.16004717e-01,1.75718278e-01,-2.39945829e-01,-1.38446122e-01,-2.48119488e-01,
1.66434005e-01,1.53335407e-01,1.15778930e-01,-1.95533589e-01,-1.51079625e-01,
2.00281575e-01,-3.02591473e-01,-7.18051270e-02,-2.35480651e-01,-2.71236487e-02,
-3.41575384e-01,-5.95962778e-02,-1.42940983e-01,-3.57108451e-02,-1.84236184e-01,
-2.08470106e-01,1.33993840e-02,-2.30474900e-02,-3.21755350e-01,-4.07095104e-01,
-4.36087161e-01,1.43216655e-01,-1.19595081e-01,2.79742517e-02,-6.75986931e-02,
-1.31325111e-01,-1.76958278e-01,-1.91665620e-01,-4.62258011e-02,2.85104841e-01,
-1.76970121e-02,1.48080826e-01,-2.09056616e-01,-4.63858657e-02,7.14390203e-02,
-7.40279675e-01,4.60031003e-01,-9.75634083e-02,-1.09208643e-01,1.90170720e-01,
-5.56358337e-01,-3.28025877e-01,1.62570357e-01,1.31312963e-02,2.16153905e-01,
-4.46718991e-01,-1.68498367e-01,-3.35785955e-01,-9.20998037e-01,1.88152209e-01,
4.77184474e-01,2.46213712e-02,3.76465440e-01,5.43788485e-02,-1.20826435e+00,
3.62382978e-02,1.35983422e-01,-3.10151696e-01,-6.64601699e-02,-1.59301794e+00,
5.36633372e-01,-6.42861426e-02,-4.52806532e-01,3.48038375e-01,-1.24470793e-01,
8.77889693e-02,7.09854737e-02,2.13614374e-01,1.26330748e-01,1.16970353e-01,
2.47931823e-01,5.42550646e-02,-5.26226163e-02,1.92816153e-01,-5.94589338e-02,
6.77908398e-03,-2.87213605e-02,1.94944933e-01,-2.36937150e-01,-3.42784002e-02,
-2.28285342e-01,1.08709827e-01,4.23206180e-01,-2.60139734e-01,-2.32469037e-01,
-1.11732721e+00,1.11836813e-01,-1.70066327e-01,1.84832722e-01,-3.87729436e-01,
-1.48297876e-01,1.78003982e-01,1.07642496e-03,-7.23514140e-01,1.38696805e-01,
-1.60401747e-01,-1.79428026e-01,1.52039140e-01,-1.03144579e-01,-6.58938065e-02,
2.57955194e-01,-3.30362320e-01,-3.82088035e-01,-1.09408097e-02,-7.43985057e-01,
7.69986734e-02,}; 
k2c_tensor conv1d_2_kernel = {&conv1d_2_kernel_array[0],3,256,{4,8,8,1,1}}; 
float conv1d_2_bias_array[8] = {
-2.70912461e-02,-2.98888795e-03,-3.36098634e-02,-5.87821268e-02,-1.24632269e-02,
-2.54034735e-02,-5.36178686e-02,4.88636568e-02,}; 
k2c_tensor conv1d_2_bias = {&conv1d_2_bias_array[0],1,8,{8,1,1,1,1}}; 

 
size_t conv1d_7_stride = 1; 
size_t conv1d_7_dilation = 1; 
float conv1d_7_output_array[264] = {0}; 
k2c_tensor conv1d_7_output = {&conv1d_7_output_array[0],2,264,{33, 8, 1, 1, 1}}; 
float conv1d_7_padded_input_array[288] = {0}; 
k2c_tensor conv1d_7_padded_input = {&conv1d_7_padded_input_array[0],2,288,{36, 8, 1, 1, 1}}; 
size_t conv1d_7_pad[2] = {1,2}; 
float conv1d_7_fill = 0.0f; 
float conv1d_7_kernel_array[256] = {
1.99029371e-01,-8.91393423e-01,-4.20627370e-02,6.12461679e-02,-2.07223803e-01,
9.23532620e-02,1.61071450e-01,4.29385602e-01,2.77937174e-01,1.70324326e-01,
2.71000803e-01,-1.38658836e-01,4.99997102e-02,-5.17818391e-01,3.04107100e-01,
6.25276417e-02,6.78318620e-01,-2.78561473e-01,-1.93292186e-01,3.03890347e-01,
1.93715155e-01,5.24982393e-01,-2.98687786e-01,6.07308522e-02,2.75633752e-01,
-4.28264529e-01,1.30235791e-01,-1.28449798e-01,1.47654280e-01,2.93084502e-01,
-6.55876040e-01,7.60715082e-02,3.29182714e-01,1.11471988e-01,3.81725840e-02,
1.63779870e-01,3.97650972e-02,9.88270640e-02,-1.69181734e-01,-3.00304085e-01,
1.65399343e-01,-1.89244539e-01,7.50314593e-02,8.35703611e-02,5.15923053e-02,
-6.18579388e-01,-3.68991822e-01,-2.81529188e-01,-2.61208694e-02,1.42134326e-02,
-1.79896709e-02,-1.13639474e-01,-3.30389440e-02,-1.12972163e-01,-1.11578274e+00,
-3.23371142e-01,9.36717838e-02,5.87988496e-02,-1.43866792e-01,1.50779292e-01,
1.12664640e-01,-4.50644195e-02,-1.16426885e-01,3.19898039e-01,-3.24949473e-01,
2.74776399e-01,-1.30274341e-01,2.52662838e-01,1.49648800e-01,2.52484977e-01,
6.54216290e-01,3.27535011e-02,1.30866826e-01,-1.36018291e-01,3.03387880e-01,
8.38008076e-02,-3.25246044e-02,-6.26382232e-01,1.41688854e-01,-2.84084201e-01,
-4.40749675e-01,1.11444369e-01,2.10646063e-01,-9.44397077e-02,-2.58671522e-01,
-1.76468536e-01,-2.57934749e-01,-1.25460088e-01,-2.78858215e-01,3.33770931e-01,
7.50912055e-02,-4.28189874e-01,2.09117364e-02,-1.07254587e-01,-1.42942384e-01,
2.42261857e-01,2.24795908e-01,-3.08773160e-01,-1.52473435e-01,-2.38386527e-01,
-2.58208513e-01,-1.19406545e+00,-3.10064465e-01,-2.92211082e-02,2.76209623e-01,
9.24063101e-02,1.66916028e-01,2.25512236e-01,-2.96568908e-02,-7.01310039e-01,
-5.03479540e-01,-3.58085304e-01,2.54294574e-01,-2.24126115e-01,-3.56626153e-01,
-2.12735027e-01,9.93600488e-03,-6.40494943e-01,-4.49365556e-01,-3.63332778e-01,
1.47150710e-01,1.11007661e-01,-9.54846069e-02,5.96002415e-02,3.06795150e-01,
2.31961131e-01,-3.62055093e-01,1.27716184e-01,-2.70237327e-02,4.82519895e-01,
2.80800551e-01,-5.70660969e-03,-5.27843945e-02,4.22088236e-01,1.69481114e-01,
4.24886286e-01,-7.51258552e-01,-4.99661058e-01,1.04631275e-01,2.43681639e-01,
-1.11571155e-01,-8.05832803e-01,2.78474927e-01,-1.12033755e-01,-2.15079501e-01,
7.57202327e-01,-2.12857753e-01,-5.35166323e-01,4.67839062e-01,9.94180590e-02,
-7.99557120e-02,3.35434049e-01,1.63763672e-01,1.47812767e-02,-1.48284286e-01,
-4.92044359e-01,1.29824713e-01,-1.05972543e-01,-5.50215989e-02,-1.05621152e-01,
-1.54652283e-01,-3.98573019e-02,-2.21706927e-01,-1.19449317e-01,-3.14812139e-02,
-6.50944054e-01,1.58495471e-01,-1.21838152e-01,-1.37840047e-01,-3.81336510e-01,
1.14912160e-01,2.67259121e-01,3.60534340e-01,-9.19848979e-01,-2.03727096e-01,
3.69926542e-01,-8.07538629e-02,-2.56886989e-01,-3.95603150e-01,-2.67572522e-01,
-3.47705841e-01,-1.55180410e-01,-2.75430501e-01,-1.84445396e-01,4.19419557e-02,
-1.25371635e-01,2.67045915e-01,1.51604548e-01,-3.23583275e-01,3.44019562e-01,
-4.24833715e-01,-2.62995269e-02,1.07605970e-02,-4.07462537e-01,2.75065541e-01,
-2.51478434e-01,-8.21415722e-01,-1.50246024e-01,3.44895795e-02,-9.13781207e-03,
-5.47305822e-01,-5.29457748e-01,-8.55841190e-02,2.35240042e-01,-2.18604088e-01,
-1.33577093e-01,6.50870025e-01,1.65381029e-01,-4.09037203e-01,-1.15852642e+00,
-3.98314536e-01,-4.55557555e-01,6.58036709e-01,-1.92552850e-01,6.51475936e-02,
5.29644489e-01,6.91820830e-02,3.75279248e-01,-6.17734194e-01,-3.39718729e-01,
1.89119577e-01,5.53111613e-01,3.79647255e-01,3.87210548e-01,2.45067254e-01,
-7.41064489e-01,1.20776311e-01,1.73659902e-02,-6.25053763e-01,-1.10924929e-01,
-1.90880015e-01,-3.65606695e-01,-2.83346474e-01,-5.51099896e-01,2.76918709e-01,
2.33792126e-01,1.37456566e-01,3.63833904e-01,-3.84348184e-01,-4.04471487e-01,
-5.49895391e-02,-3.44198316e-01,-2.41877377e-01,-2.04820514e-01,-4.84485209e-01,
1.01682497e-02,-7.50736713e-01,-1.45450205e-01,-3.24032187e-01,2.27110296e-01,
-4.08437997e-01,-3.84368837e-01,3.48431617e-02,9.49472114e-02,9.55692902e-02,
5.49938008e-02,}; 
k2c_tensor conv1d_7_kernel = {&conv1d_7_kernel_array[0],3,256,{4,8,8,1,1}}; 
float conv1d_7_bias_array[8] = {
-1.07119195e-01,-1.47512525e-01,-6.49717450e-03,-4.17849757e-02,-6.25185966e-02,
-1.70700282e-01,1.82382632e-02,-5.26063181e-02,}; 
k2c_tensor conv1d_7_bias = {&conv1d_7_bias_array[0],1,8,{8,1,1,1,1}}; 

 
size_t conv1d_3_stride = 1; 
size_t conv1d_3_dilation = 1; 
float conv1d_3_output_array[264] = {0}; 
k2c_tensor conv1d_3_output = {&conv1d_3_output_array[0],2,264,{33, 8, 1, 1, 1}}; 
float conv1d_3_padded_input_array[320] = {0}; 
k2c_tensor conv1d_3_padded_input = {&conv1d_3_padded_input_array[0],2,320,{40, 8, 1, 1, 1}}; 
size_t conv1d_3_pad[2] = {3,4}; 
float conv1d_3_fill = 0.0f; 
float conv1d_3_kernel_array[512] = {
-5.26012003e-01,-5.93549490e-01,3.65499556e-01,5.49072109e-04,-3.63098592e-01,
2.10637122e-01,-6.43720701e-02,9.37806964e-02,5.06692469e-01,-7.70819724e-01,
5.04905462e-01,-6.23380113e-03,-4.03066754e-01,-1.11205354e-01,2.07142651e-01,
-1.42375574e-01,-1.50380090e-01,-5.09366870e-01,1.43596306e-01,-9.86644998e-03,
-1.67545825e-01,9.36113223e-02,1.84952497e-01,-2.84142848e-02,3.09835583e-01,
-1.21571398e+00,4.04878020e-01,1.72644199e-04,-3.14047962e-01,1.30104974e-01,
1.37276202e-01,2.95700192e-01,2.25019887e-01,3.56570721e-01,-6.38280392e-01,
6.41816761e-03,-2.21957818e-01,-2.43157819e-01,1.07653867e-02,-1.93409532e-01,
-3.20641845e-01,-4.89278913e-01,5.28097190e-02,-8.77559185e-03,-4.65213537e-01,
-1.62071228e-01,5.98292723e-02,-1.77763045e-01,3.55406046e-01,-3.67560595e-01,
4.69922543e-01,-3.06406827e-03,-6.86868504e-02,1.67451262e-01,-6.54574111e-02,
1.24994099e-01,-2.64711589e-01,1.38589352e-01,-4.60818797e-01,3.59382620e-03,
-2.25906506e-01,-9.20609593e-01,-1.75419807e-01,1.10154934e-01,-5.64338803e-01,
1.72144353e-01,8.23053792e-02,-2.75274483e-03,-7.27662623e-01,-4.88202684e-02,
1.12870894e-01,1.89965144e-02,4.41523880e-01,4.08804044e-02,1.62697613e-01,
3.43656284e-03,-4.41542923e-01,-9.60046574e-02,1.57870173e-01,-7.83306509e-02,
-1.86020195e-01,-4.56500202e-01,5.13457917e-02,6.59445999e-03,-2.07860991e-01,
3.97654027e-01,2.72817373e-01,1.24902606e-01,1.30450174e-01,-3.36137682e-01,
2.49613002e-01,-9.18144186e-04,-8.25335905e-02,1.70516893e-02,-1.47271425e-01,
6.34411350e-02,-1.47847040e-02,2.70119369e-01,-5.28246403e-01,-9.99653339e-03,
1.52057797e-01,-3.98581892e-01,-1.58702612e-01,1.19914316e-01,-5.58752477e-01,
-1.94136679e-01,1.52758792e-01,-5.68131870e-03,5.58953993e-02,2.85211325e-01,
-1.28788417e-02,-1.17138818e-01,4.74117994e-01,-2.81046063e-01,2.63592809e-01,
6.83417218e-03,3.88429947e-02,-2.20446229e-01,-2.18346506e-01,-2.13959254e-02,
1.33137912e-01,3.28329504e-01,-6.05595589e-01,-4.31875605e-03,3.02504059e-02,
-5.83673179e-01,-1.80890575e-01,1.79664090e-01,-1.23562827e-03,-4.31378260e-02,
3.72260600e-01,1.18776679e-03,-6.69540346e-01,9.97018591e-02,6.16147816e-02,
8.16349611e-02,-1.08041622e-01,-1.59368291e-01,9.00347531e-02,-6.70961896e-03,
-7.53564596e-01,8.24511945e-02,-1.48543283e-01,2.03004926e-01,-5.30322492e-02,
-6.39707208e-01,1.06669046e-01,-1.30944550e-02,-1.12485297e-01,3.68979067e-01,
2.99383700e-01,-4.62221280e-02,1.85817853e-02,-5.12179375e-01,2.30714932e-01,
6.29157946e-03,-1.20232977e-01,7.35509619e-02,-1.06066160e-01,-1.21704005e-01,
1.02310039e-01,1.17787294e-01,-4.87152964e-01,1.13137783e-02,3.10879260e-01,
-6.14346564e-01,-1.48608878e-01,-1.66639596e-01,-3.12322676e-01,-1.99417755e-01,
-1.61252424e-01,-2.14059348e-03,-3.77381057e-01,-1.15005985e-01,6.81319982e-02,
-4.19441722e-02,5.99617898e-01,-4.92943078e-01,5.29121816e-01,-4.46766680e-05,
-1.62130848e-01,3.13065886e-01,1.00579843e-01,2.06469614e-02,9.02938377e-03,
1.01549819e-01,-4.63871211e-01,1.26398625e-02,6.67414069e-02,-7.06766069e-01,
-2.17053339e-01,2.80888736e-01,-1.69294283e-01,-4.73858118e-02,2.09120303e-01,
2.49675754e-02,-2.08944842e-01,-7.38477185e-02,2.45050609e-01,-7.17775077e-02,
-1.00236498e-01,2.78426766e-01,5.12292869e-02,-2.85548507e-04,-1.02508330e+00,
2.21927926e-01,8.69524479e-02,2.57774532e-01,7.25137368e-02,-1.42372465e+00,
1.24761853e-02,-4.06304980e-03,-1.83993399e-01,1.97519377e-01,-1.07402347e-01,
-6.03816891e-03,1.22830130e-01,-5.28287888e-01,1.85998216e-01,1.23296529e-02,
5.64683788e-02,1.96987465e-01,-5.23911566e-02,8.12628418e-02,-2.64605880e-01,
1.69491991e-01,-4.48616266e-01,-1.81580856e-02,2.29219094e-01,-3.47289890e-01,
-1.26473084e-01,-7.15362430e-02,-7.83812284e-01,-8.22203338e-01,-2.41350099e-01,
4.77398251e-04,-4.24770892e-01,-4.80160922e-01,5.30454591e-02,2.22911891e-02,
7.45308876e-01,-1.00970767e-01,2.22650051e-01,-1.37696753e-03,1.08952671e-01,
3.39000791e-01,2.03758270e-01,-1.68681219e-02,-6.72799647e-02,5.70878610e-02,
-4.38102096e-01,3.16700488e-02,9.30908471e-02,-4.73444879e-01,-1.88160434e-01,
-2.42417529e-02,3.67726833e-02,3.36085975e-01,1.36009470e-01,-9.53987183e-05,
2.04467811e-02,-9.68927965e-02,1.15784958e-01,2.18963623e-01,3.03215414e-01,
-4.89242449e-02,1.21835679e-01,3.67437047e-03,-9.02393043e-01,3.88095438e-01,
-2.33065248e-01,-1.57444403e-01,1.04943819e-01,-7.57482946e-01,1.22142680e-01,
-2.99399742e-03,-5.79788625e-01,9.43635311e-03,-1.21117212e-01,-3.57244839e-03,
8.55692402e-02,-6.94296896e-01,1.40040562e-01,-3.10304505e-03,-1.21508345e-01,
-8.35624465e-04,-2.44414508e-01,1.15396813e-01,-3.76788914e-01,3.62210721e-01,
-3.17022234e-01,-1.16165366e-03,2.68012881e-01,-1.28390148e-01,-2.40684226e-01,
-1.52522428e-02,-3.95934016e-01,1.50953725e-01,-5.12754954e-02,-6.91501005e-03,
1.30693242e-01,1.76337957e-02,-3.94358970e-02,-1.13030739e-01,7.01299608e-01,
-2.51340747e-01,1.53927535e-01,4.27329331e-04,-1.39268696e-01,8.49204510e-02,
1.27827540e-01,-4.16359417e-02,-7.85261512e-01,-4.23363671e-02,-2.01983169e-01,
2.93461420e-03,-1.73406601e-01,9.52218622e-02,-4.13787961e-01,1.65898930e-02,
-8.14257115e-02,1.56912997e-01,1.68277860e-01,-1.17687357e-03,-7.01354086e-01,
-8.12045112e-02,2.95935929e-01,-3.42689408e-03,4.94207561e-01,-2.88660645e-01,
3.19885164e-01,2.77809286e-03,-1.55523092e-01,2.22820655e-01,5.46731651e-02,
-1.70779988e-01,2.87541926e-01,-5.01573563e-01,2.07257234e-02,1.46383827e-03,
-9.61117983e-01,-3.81226450e-01,1.41042054e-01,-8.31903070e-02,2.55511820e-01,
-4.56905633e-01,-2.28307629e-03,-4.78788599e-04,-6.50997639e-01,-4.89356518e-02,
-1.74342290e-01,8.15556664e-03,-5.06861925e-01,1.47783190e-01,-4.07661021e-01,
-1.21808425e-03,1.55071214e-01,-5.17963111e-01,-1.60870180e-01,8.13924819e-02,
3.32441270e-01,5.46761043e-03,1.93512738e-01,6.60238462e-03,-2.62051821e-01,
-1.22098908e-01,5.51842637e-02,-3.04482967e-01,9.10501704e-02,-3.65611255e-01,
5.67509048e-02,-4.36757924e-03,-3.01480442e-01,6.38197720e-01,2.00925678e-01,
1.92965001e-01,-9.50943828e-01,3.64710420e-01,-3.00650239e-01,1.46776612e-03,
-5.15468605e-02,-6.09985963e-02,-1.49422020e-01,-5.91156222e-02,2.83805188e-03,
2.16138229e-01,2.98303664e-01,4.60893475e-03,-1.02130055e+00,6.04323205e-03,
2.87904173e-01,-3.46475430e-02,-2.30099425e-01,-2.56212503e-01,3.82520884e-01,
4.35695890e-03,-1.82969624e-03,1.68144390e-01,1.52406201e-01,-1.13242626e-01,
4.93538141e-01,-1.02385804e-01,-1.94599293e-02,-2.56244675e-03,-8.76387119e-01,
-4.09645945e-01,1.77049220e-01,-3.28434035e-02,1.75555214e-01,-5.74886024e-01,
-1.04710989e-01,7.29114050e-04,-1.18800962e+00,1.70215398e-01,-3.95841263e-02,
-8.66146684e-02,-5.79867065e-01,2.30530456e-01,-7.22601652e-01,-1.80800969e-03,
2.96974212e-01,-5.11827707e-01,-2.05457747e-01,1.43145964e-01,1.99155957e-01,
-4.50920552e-01,2.76664078e-01,-6.57900237e-04,-1.23595215e-01,-6.20633602e-01,
8.24313313e-02,-2.93649435e-02,5.70409954e-01,6.37740374e-01,4.46479797e-01,
5.13772294e-03,2.44669944e-01,6.39916003e-01,3.91333252e-01,1.50914207e-01,
-1.25058544e+00,4.51962166e-02,-1.75712574e-02,-4.03197220e-04,1.70043662e-01,
-3.95718180e-02,-5.98741882e-02,-4.55282569e-01,1.83988679e-02,-1.11857951e-02,
5.35417140e-01,4.05143993e-03,-7.30970562e-01,6.04941964e-01,1.08861633e-01,
5.58525212e-02,-3.06417435e-01,-3.53959262e-01,4.14389551e-01,3.61281773e-03,
-5.51458955e-01,-3.98051381e-01,-6.05580071e-03,-4.54019569e-02,5.13745546e-01,
-7.00169265e-01,4.48117331e-02,2.16921396e-03,-8.79484951e-01,-1.81722209e-01,
-1.21341221e-01,-6.50802851e-02,1.87219858e-01,-9.87270355e-01,1.48028925e-01,
3.57878814e-03,-1.21145868e+00,4.92403001e-01,-1.65028766e-01,-2.06026822e-01,
-7.03364909e-01,5.29036820e-01,-5.08435786e-01,-6.54234551e-03,6.09915912e-01,
-3.86184216e-01,-1.28423691e-01,-2.26009965e-01,-6.69657588e-02,3.34158689e-02,
6.59114718e-02,4.12729150e-03,-2.63392150e-01,-6.69991672e-01,1.25746012e-01,
4.90562357e-02,6.01053119e-01,-4.87785321e-03,5.30212581e-01,-2.03368196e-04,
-2.77035199e-02,6.59493089e-01,1.32767618e-01,-9.38339308e-02,-1.07101476e+00,
-9.64574069e-02,-3.87975015e-02,9.29161208e-04,3.38237047e-01,-7.12136328e-01,
-9.78953540e-02,-2.18682289e-01,}; 
k2c_tensor conv1d_3_kernel = {&conv1d_3_kernel_array[0],3,512,{8,8,8,1,1}}; 
float conv1d_3_bias_array[8] = {
-2.26819366e-01,-1.02922350e-01,-2.43505031e-01,5.70469128e-04,-6.62089214e-02,
-1.53092757e-01,-2.59878188e-01,-5.22539802e-02,}; 
k2c_tensor conv1d_3_bias = {&conv1d_3_bias_array[0],1,8,{8,1,1,1,1}}; 

 
size_t conv1d_8_stride = 1; 
size_t conv1d_8_dilation = 1; 
float conv1d_8_output_array[264] = {0}; 
k2c_tensor conv1d_8_output = {&conv1d_8_output_array[0],2,264,{33, 8, 1, 1, 1}}; 
float conv1d_8_padded_input_array[320] = {0}; 
k2c_tensor conv1d_8_padded_input = {&conv1d_8_padded_input_array[0],2,320,{40, 8, 1, 1, 1}}; 
size_t conv1d_8_pad[2] = {3,4}; 
float conv1d_8_fill = 0.0f; 
float conv1d_8_kernel_array[512] = {
-1.74987093e-01,-1.04739174e-01,-3.97079512e-02,-2.22744986e-01,-1.35294512e-01,
-2.78621256e-01,-1.39636934e-01,-3.73427600e-01,-5.66043913e-01,4.67975229e-01,
2.08242983e-02,1.62340611e-01,3.56134586e-02,-1.03005521e-01,-9.52993482e-02,
-2.42611527e-01,-2.11748555e-01,2.42728949e-01,5.14279045e-02,6.71713710e-01,
1.30926222e-01,3.90096754e-01,3.35596085e-01,-8.71752977e-01,4.29963022e-01,
1.20720074e-01,9.72126797e-02,1.54202297e-01,1.87938169e-01,3.76625508e-01,
1.30971149e-02,-5.54040134e-01,-4.98263896e-01,6.89727291e-02,1.40066788e-01,
-4.20906991e-01,3.53857934e-01,3.53471749e-02,-3.41281414e-01,-3.83866727e-01,
2.28957713e-01,-1.48242628e-02,-7.90591165e-02,-1.36407390e-01,1.36443814e-02,
4.80093241e-01,-6.05276883e-01,6.51424408e-01,-7.98645020e-02,-1.31201476e-01,
-1.33045660e-02,5.89689240e-03,-1.37310728e-01,1.17049031e-01,6.23492524e-02,
-7.94186294e-01,-4.17479545e-01,-1.34842917e-01,-5.80505282e-02,-5.48208773e-01,
-7.84879178e-02,-1.92354023e-01,1.80829972e-01,4.99579310e-01,-2.35260546e-01,
-1.50270566e-01,-2.28079688e-02,-4.62874621e-01,6.16755448e-02,-1.85739353e-01,
-4.10493433e-01,-2.73665488e-01,4.80111748e-01,-9.73831341e-02,3.92549373e-02,
-4.41605449e-02,7.89622366e-02,-1.85141966e-01,9.37777907e-02,1.91045433e-01,
-1.11942239e-01,3.05413753e-01,3.11859660e-02,4.86772716e-01,2.45927311e-02,
2.74094790e-01,-7.73467794e-02,-3.56761329e-02,3.19697529e-01,-3.97484526e-02,
-3.64685170e-02,8.59240294e-02,8.62681866e-03,2.57097781e-01,1.29359916e-01,
-2.50513196e-01,-3.30253959e-01,-8.83519370e-03,2.02589139e-01,-3.41751188e-01,
4.18700725e-01,-3.08038265e-01,1.14771957e-02,3.10583562e-01,2.07072884e-01,
1.54841438e-01,9.45234746e-02,-4.74006236e-02,-2.61333793e-01,5.16246855e-01,
1.52570009e-01,3.60431463e-01,-3.64919364e-01,-1.76088244e-01,-5.00785597e-02,
-3.92092735e-01,2.01483414e-01,2.99013294e-02,2.67854985e-02,4.43301529e-01,
-1.85221627e-01,-5.58906160e-02,-4.67558317e-02,-2.04838768e-01,-2.17513427e-01,
-1.13619514e-01,-1.10199511e-01,6.24967739e-02,-2.82402545e-01,-2.39556476e-01,
-2.44383272e-02,-4.62517202e-01,-1.09341089e-02,-3.47830325e-01,-2.55475730e-01,
-6.40258133e-01,-3.68978143e-01,2.98119545e-01,-2.74133105e-02,2.03988343e-01,
1.10318735e-01,-4.19503480e-01,1.42457843e-01,-5.12478724e-02,1.76778913e-01,
3.07144612e-01,1.09256625e-01,6.15401566e-01,1.44134924e-01,2.62181848e-01,
3.67468059e-01,-6.95557356e-01,1.51537403e-01,8.00416395e-02,4.86796536e-02,
-1.04760058e-01,9.94407292e-03,4.02683347e-01,5.69734573e-02,-5.06120801e-01,
1.34049773e-01,1.46431819e-01,1.20318800e-01,-2.31450960e-01,2.30318367e-01,
-1.46840781e-01,-6.41137082e-03,-2.30337493e-02,8.22518766e-01,-3.08967233e-01,
-4.80831750e-02,7.68683702e-02,-2.08656847e-01,3.98539782e-01,-4.40978020e-01,
1.91580966e-01,-2.60191038e-02,-1.89339370e-02,4.89002988e-02,-3.97982448e-01,
3.48686315e-02,-5.71292825e-02,-1.26509950e-01,3.12196851e-01,6.96670935e-02,
-8.96629542e-02,-4.40213010e-02,7.04559162e-02,-2.24581540e-01,-9.05791372e-02,
1.81550309e-01,-2.80729849e-02,-1.23743214e-01,-2.48259962e-01,-1.47278737e-02,
-3.87548178e-01,3.63610499e-02,-4.14925724e-01,-2.23859727e-01,-1.45741314e-01,
4.56493944e-01,1.57409683e-01,-1.53605103e-01,-1.92317307e-01,3.04965883e-01,
-3.36190462e-01,-8.14485624e-02,1.68195173e-01,1.51433066e-01,3.00540209e-01,
1.12679955e-02,3.61634612e-01,1.93943791e-02,2.33130276e-01,3.94848347e-01,
-1.36420977e+00,2.45547578e-01,2.64644235e-01,2.71084458e-02,-5.89292169e-01,
2.56558925e-01,5.02146959e-01,2.12500662e-01,-6.55962110e-01,-1.20972097e-01,
1.83582217e-01,1.04652718e-02,-2.51773894e-01,4.66571599e-01,-1.69267043e-01,
-2.16915324e-01,7.20923021e-02,3.02663624e-01,-2.37726182e-01,1.59532979e-01,
-1.52540550e-01,-5.14091253e-01,9.02493112e-03,-2.26899832e-02,5.47108948e-01,
-1.68186173e-01,-9.82364863e-02,5.02944626e-02,-2.13712439e-01,7.33249933e-02,
-8.33481550e-03,-1.73731446e-01,1.49004623e-01,7.37271309e-02,-1.55844450e-01,
8.05276856e-02,-6.37956381e-01,-2.31554493e-01,-9.34845656e-02,-1.91024144e-03,
-3.10656071e-01,-3.10442392e-02,-3.05518359e-01,-4.41301102e-03,-4.26053889e-02,
2.81359833e-02,-1.42718658e-01,-7.46740460e-01,-2.56470591e-01,5.24009585e-01,
5.43163978e-02,-8.78036842e-02,1.54676780e-01,1.57141536e-01,-7.13550568e-01,
3.49773943e-01,4.85146582e-01,-1.02068290e-01,1.56825602e-01,5.15406802e-02,
-6.10364135e-03,3.28010708e-01,3.25763047e-01,-1.57587036e-01,-9.03018296e-01,
3.71744007e-01,2.96214193e-01,-2.42245570e-02,3.42774004e-01,-9.42426547e-02,
4.24191654e-01,-2.71839410e-01,1.50963306e-01,1.22155689e-01,-5.32652661e-02,
-4.12189737e-02,-5.32222867e-01,1.19431525e-01,-2.14665532e-01,-1.35233551e-01,
2.04973862e-01,4.09360193e-02,-2.31438607e-01,1.74038783e-01,-1.03130363e-01,
-6.62634194e-01,6.07826635e-02,4.22651283e-02,6.50006711e-01,2.20472127e-01,
-3.42619941e-02,7.56949782e-02,1.69133261e-01,3.41065377e-01,-1.61179230e-01,
9.64763761e-02,9.92212817e-02,-2.50337571e-01,-2.24414453e-01,1.08251907e-01,
-8.69895697e-01,-1.03053279e-01,-5.47532402e-02,1.33804604e-01,-1.98585093e-01,
-3.29348534e-01,-1.22105777e-01,4.78334306e-03,1.87454566e-01,-4.76760752e-02,
-7.33691573e-01,-1.77139252e-01,-6.12023830e-01,6.18077576e-01,-9.17901173e-02,
-7.85545334e-02,9.25231539e-03,2.46891886e-01,-2.39793941e-01,-6.05206072e-01,
1.49586618e-01,-1.24472968e-01,2.59267330e-01,-4.08183709e-02,-5.61870396e-01,
-1.12040587e-01,2.95656294e-01,-9.33464840e-02,-5.40066063e-01,2.99775213e-01,
2.61097163e-01,-3.60709764e-02,5.58193564e-01,-2.99437940e-01,2.67587215e-01,
-4.02868301e-01,-9.12678778e-01,6.44281283e-02,-1.84006631e-01,-1.33195743e-01,
-3.33500892e-01,1.72220916e-01,-1.33442447e-01,-2.55556792e-01,3.75498146e-01,
5.84358454e-01,-2.97893435e-01,1.96350031e-02,-8.45827609e-02,-3.92764658e-01,
8.61013308e-02,4.82847720e-01,4.47333246e-01,-7.23215118e-02,-1.08427979e-01,
-6.08527735e-02,-4.32303041e-01,3.11733689e-02,-1.43689886e-01,-1.33188635e-01,
8.75106081e-03,-7.59825930e-02,-1.96044743e-01,1.60674065e-01,-7.67481506e-01,
-1.28051609e-01,-2.50640754e-02,5.21820523e-02,-2.05592468e-01,-8.10420737e-02,
-2.23311894e-02,3.64102307e-03,1.10835537e-01,-3.39483395e-02,-3.03028468e-02,
-3.83559227e-01,-3.16565245e-01,2.29847208e-01,-9.00396258e-02,-4.62834053e-02,
-7.44907781e-02,2.28817552e-01,-5.78817070e-01,1.24518625e-01,4.20649976e-01,
-9.00065824e-02,1.59865379e-01,5.76185770e-02,1.44306859e-02,1.58698380e-01,
1.69918910e-01,1.36845008e-01,-8.65257740e-01,5.50566196e-01,2.93558776e-01,
-2.77622249e-02,4.22602445e-01,1.43177703e-01,3.01317722e-01,-4.88646328e-01,
2.05248758e-01,1.58736855e-01,-4.06022489e-01,-4.94413711e-02,-5.31065166e-01,
3.04472595e-01,-5.87195903e-02,5.90175390e-02,1.35596037e-01,1.51004598e-01,
1.10305227e-01,2.82861173e-01,7.44034424e-02,-5.00949681e-01,1.29501358e-01,
-4.52836931e-01,3.61197561e-01,-6.54881224e-02,-5.67771792e-02,2.39323173e-02,
-1.60236116e-02,1.66189373e-01,-9.93772298e-02,-1.56376716e-02,-2.30261683e-02,
1.50174305e-01,-2.51081109e-01,8.47664550e-02,-5.87976456e-01,-1.33559689e-01,
-2.63379235e-02,-2.65374601e-01,-1.85529843e-01,-4.15720135e-01,-8.51829201e-02,
-1.75626557e-02,-2.64330417e-01,5.74094243e-02,-1.77140743e-01,1.58494160e-01,
-7.04489052e-02,-9.07386780e-01,-2.99209133e-02,-2.75484640e-02,1.50233163e-02,
1.92958442e-03,-2.25799978e-01,2.01010779e-01,3.00325453e-01,2.58196801e-01,
2.74327040e-01,-2.33241338e-02,-5.53317368e-01,-5.67450672e-02,1.90413430e-01,
-1.17946781e-01,-1.30070341e+00,4.52716768e-01,8.37278068e-01,-1.41473100e-01,
-2.44508192e-01,3.27006906e-01,3.05919111e-01,3.70160192e-01,3.93246323e-01,
-3.63898575e-02,-4.51225549e-01,-1.24258839e-01,-3.14724356e-01,2.05719650e-01,
-7.91420788e-02,3.81490914e-03,-2.80792359e-03,-7.14208543e-01,-8.20127428e-02,
3.91461253e-01,-9.26098153e-02,-6.91865742e-01,4.09478337e-01,-2.67607570e-01,
7.06231058e-01,2.25244060e-01,-2.57453352e-01,-8.74238312e-02,-1.95201322e-01,
8.76291171e-02,-2.00778656e-02,-8.73842165e-02,2.00490698e-01,1.43945098e-01,
-4.47395325e-01,1.38933063e-01,-5.32680929e-01,-3.25304598e-01,-6.13396689e-02,
3.81730385e-02,-5.09677269e-02,}; 
k2c_tensor conv1d_8_kernel = {&conv1d_8_kernel_array[0],3,512,{8,8,8,1,1}}; 
float conv1d_8_bias_array[8] = {
-3.94203067e-02,-1.23452209e-01,4.16282155e-02,-1.94899872e-01,9.46302265e-02,
-1.25329897e-01,-1.25677630e-01,-1.14102304e-01,}; 
k2c_tensor conv1d_8_bias = {&conv1d_8_bias_array[0],1,8,{8,1,1,1,1}}; 

 
size_t concatenate_5_num_tensors0 = 3; 
size_t concatenate_5_axis = -2; 
float concatenate_5_output_array[792] = {0}; 
k2c_tensor concatenate_5_output = {&concatenate_5_output_array[0],2,792,{33,24, 1, 1, 1}}; 


size_t concatenate_6_num_tensors0 = 3; 
size_t concatenate_6_axis = -2; 
float concatenate_6_output_array[792] = {0}; 
k2c_tensor concatenate_6_output = {&concatenate_6_output_array[0],2,792,{33,24, 1, 1, 1}}; 


size_t conv1d_4_stride = 1; 
size_t conv1d_4_dilation = 1; 
float conv1d_4_output_array[330] = {0}; 
k2c_tensor conv1d_4_output = {&conv1d_4_output_array[0],2,330,{33,10, 1, 1, 1}}; 
float conv1d_4_padded_input_array[864] = {0}; 
k2c_tensor conv1d_4_padded_input = {&conv1d_4_padded_input_array[0],2,864,{36,24, 1, 1, 1}}; 
size_t conv1d_4_pad[2] = {1,2}; 
float conv1d_4_fill = 0.0f; 
float conv1d_4_kernel_array[960] = {
-5.61347827e-02,3.33714345e-03,-1.23649985e-01,5.28468676e-02,-1.61923662e-01,
3.40547152e-02,-8.22306946e-02,1.13744475e-01,-2.94328965e-02,-2.70774737e-02,
-2.05613181e-01,-1.02292359e-01,1.54702943e-02,3.16570736e-02,-2.49578264e-02,
-8.82848054e-02,3.28940228e-02,2.15271235e-01,1.30495161e-01,1.02761246e-01,
-6.95951656e-02,1.31509453e-02,2.46998332e-02,-2.72591710e-02,5.70522472e-02,
9.12139425e-04,5.77325970e-02,-1.75844636e-02,3.90769392e-02,6.27768710e-02,
1.99670661e-02,-1.37461111e-01,5.51355556e-02,3.45995314e-02,6.60420209e-02,
-1.31130561e-01,-2.68555190e-02,-1.32350087e-01,-9.16989744e-02,3.63751751e-05,
1.08616091e-01,8.01932886e-02,6.94816932e-02,1.03221074e-01,-1.14500634e-02,
7.60514885e-02,6.91322535e-02,1.79309919e-02,1.42185343e-02,4.07068357e-02,
-7.94835389e-03,-4.56924662e-02,-1.15323603e-01,8.19168314e-02,-6.91547245e-02,
-4.55438606e-02,-5.38467057e-02,1.34432301e-01,-2.70300042e-02,-1.00481793e-01,
-6.95587248e-02,8.39456543e-02,-1.10091649e-01,-2.01505467e-01,8.72223079e-02,
1.12290077e-01,-5.02196662e-02,-3.50547321e-02,3.39673012e-02,5.53639270e-02,
-5.84278964e-02,-2.17524376e-02,-1.08462654e-01,-3.21665145e-02,-1.29462570e-01,
-4.64180484e-02,-9.33315530e-02,-8.01458582e-02,-8.67763907e-02,-9.84764993e-02,
-1.28263086e-01,-1.76778764e-01,2.17401441e-02,-1.91334896e-02,6.25916496e-02,
-1.21410862e-01,-1.44416228e-01,-4.72500781e-03,7.51590282e-02,1.75177481e-03,
5.05154394e-02,1.08827643e-01,-3.59341651e-02,-2.42801141e-02,1.94986984e-02,
2.19861850e-01,1.24034192e-03,-2.49656197e-02,-7.79392943e-02,-1.57835409e-01,
-6.92084208e-02,-1.95387468e-01,1.23130761e-01,-3.55589651e-02,5.14133461e-02,
-1.22939564e-01,7.44815916e-02,6.34568110e-02,-7.72509649e-02,1.10279158e-01,
1.20927580e-01,7.30861798e-02,4.34518866e-02,5.74525595e-02,7.27209402e-03,
-1.83504611e-01,1.48066849e-01,-1.45526543e-01,2.33455691e-02,1.11356795e-01,
1.23875231e-01,-5.88501571e-03,-9.22598094e-02,6.66643232e-02,-1.08775064e-01,
-3.07736751e-02,-1.14849202e-01,2.76540350e-02,-6.05424270e-02,-1.76750086e-02,
-1.61362007e-01,3.03708222e-02,-3.16143930e-02,-1.42545834e-01,-4.46664635e-03,
2.19062082e-02,3.11599094e-02,-1.03525938e-02,4.43477593e-02,1.99295714e-01,
3.36029157e-02,9.66182873e-02,-7.41880238e-02,1.15493737e-01,-1.56924292e-01,
4.72650155e-02,-5.36617339e-02,-1.58557132e-01,-7.71633387e-02,-2.60852963e-01,
4.02472839e-02,1.05480455e-01,2.70066634e-02,1.17297158e-01,7.18189925e-02,
8.81540924e-02,1.40154481e-01,3.19682658e-02,3.52407172e-02,-2.49732248e-02,
3.80258977e-01,-4.25621837e-01,-9.41209048e-02,1.23667710e-01,1.28171109e-02,
5.45546889e-01,-6.31257713e-01,-3.54367346e-01,-2.38196909e-01,9.69082177e-01,
-8.03488433e-01,5.34131825e-01,2.44846269e-01,8.35018009e-02,7.99044222e-02,
-1.76912159e-01,4.74654555e-01,-6.24432229e-03,-2.19728932e-01,-1.84056115e+00,
2.87659854e-01,-3.96891832e-01,3.02227940e-02,-8.27167183e-03,3.01075280e-02,
2.13284597e-01,8.77659172e-02,-1.80534333e-01,2.02098638e-01,1.13387191e+00,
-1.18323162e-01,2.64327735e-01,-1.41329110e-01,-2.27956727e-01,-1.50735900e-01,
7.92089477e-02,2.29081646e-01,-8.58440772e-02,-2.26506636e-01,-4.84858602e-02,
-1.68189555e-01,3.03551286e-01,-5.71386293e-02,-9.03920829e-01,-4.93919522e-01,
-3.27944547e-01,4.00338441e-01,3.26189816e-01,2.39113152e-01,-3.89541030e-01,
-8.78519043e-02,-4.51355040e-01,-2.28659753e-02,1.88661397e-01,1.98949754e-01,
3.68499964e-01,-4.80953127e-01,-3.30006033e-01,-3.04173976e-01,4.76920694e-01,
-1.91114292e-01,1.01666480e-01,1.11505147e-02,6.06679870e-03,-1.60873353e-01,
-6.19714037e-02,4.27877963e-01,8.69210288e-02,-2.99450070e-01,-5.13319790e-01,
-2.29284503e-02,6.59992769e-02,1.25207782e-01,-2.45352015e-01,-1.41539693e-01,
-7.45245889e-02,1.20406516e-01,2.98628043e-02,1.27512187e-01,1.42764589e-02,
6.15102686e-02,6.12460487e-02,-6.92341253e-02,5.85238338e-02,9.66445133e-02,
-5.97733633e-05,-1.80471987e-02,-1.79450274e-01,-2.38007512e-02,-4.42074351e-02,
3.69031243e-02,8.31535533e-02,1.23204291e-01,-3.66312228e-02,-1.72870066e-02,
-3.50302756e-02,5.85690103e-02,-7.19502792e-02,1.73713490e-01,1.40085950e-01,
6.33335412e-02,4.68202755e-02,1.56095708e-02,4.94862720e-02,-9.32517350e-02,
-5.62201701e-02,-2.29927152e-02,-6.50981143e-02,8.90040472e-02,-7.12274536e-02,
1.20454896e-02,-1.18334755e-01,1.64692134e-01,-7.54612014e-02,1.24142647e-01,
5.83340898e-02,4.77286167e-02,5.82959838e-02,-5.62633313e-02,8.18307325e-02,
-8.87408853e-02,-3.00229657e-02,-9.22448039e-02,-1.51354317e-02,-3.83424535e-02,
-7.44277388e-02,-8.99165347e-02,-1.15581319e-01,-4.05574329e-02,1.30018834e-02,
9.10059810e-02,-6.22595139e-02,6.17810823e-02,8.86460245e-02,1.32514313e-01,
1.09903075e-01,-1.43468395e-01,8.50488469e-02,-2.17541165e-05,-1.39482424e-01,
-6.93568885e-02,-7.73200467e-02,1.66853309e-01,9.19841155e-02,-9.15563405e-02,
-4.09559943e-02,-2.61395164e-02,9.96071547e-02,1.08057447e-01,4.40594889e-02,
2.25585550e-02,-7.91274291e-03,8.83136392e-02,1.91462543e-02,9.83086377e-02,
6.43958226e-02,1.23724148e-01,1.71549276e-01,1.30910754e-01,1.03974454e-01,
1.50105273e-02,-6.49101809e-02,2.35637240e-02,-1.58149805e-02,-1.02921568e-01,
-4.51380014e-03,-5.61265461e-02,-3.53624634e-02,-8.45585987e-02,-2.25507319e-02,
6.96365535e-02,2.98757777e-02,-5.95444813e-02,2.44380925e-02,1.01222098e-01,
-8.11221749e-02,5.55273145e-02,6.83906600e-02,1.35145351e-01,1.59890011e-01,
-2.35019848e-02,-1.00410454e-01,-4.86853868e-02,-5.63839115e-02,2.42360365e-02,
-1.66394979e-01,-1.57065853e-01,1.01707205e-01,1.53046802e-01,-1.27340659e-01,
-1.43223062e-01,-3.27565521e-02,-1.10320316e-03,1.09270653e-02,-1.70109019e-01,
-1.64300814e-01,-1.26749367e-01,-9.45965573e-02,8.28467086e-02,4.10317704e-02,
-4.05024141e-02,-1.02474019e-02,8.59961063e-02,-4.58208956e-02,1.29893675e-01,
-4.28101867e-02,-2.80920397e-02,-9.11428630e-02,1.44587001e-02,-9.46787149e-02,
-2.84920871e-01,9.64955986e-02,-5.83996475e-02,-1.49283066e-01,-7.14165345e-02,
7.00787678e-02,2.04137996e-01,1.80800557e-01,1.35674238e-01,1.83081493e-01,
7.98825026e-02,5.98818110e-03,-9.25876051e-02,-2.28768736e-01,1.12023197e-01,
1.69154316e-01,1.61598343e-02,-7.03617558e-02,-6.94318190e-02,1.08969711e-01,
1.31084800e-01,-3.49662751e-02,3.08300741e-02,5.35427928e-02,1.43928126e-01,
6.46037534e-02,-7.26784542e-02,-1.18032426e-01,-5.32488041e-02,-1.96079746e-01,
5.41703939e-01,-3.49929929e-01,-2.28701353e-01,8.17853883e-02,3.94948006e-01,
4.47357655e-01,-3.57521981e-01,1.10263437e-01,1.61988482e-01,4.68681514e-01,
-3.81725729e-01,3.56444031e-01,7.24952281e-01,3.60790007e-02,2.35514045e-01,
-1.49929404e-01,4.75640476e-01,-1.51202632e-02,-9.30619612e-02,-1.44828796e+00,
2.90253878e-01,-1.34935841e-01,-2.50883460e-01,-1.43553331e-01,-4.57439050e-02,
-2.36113127e-02,6.77266493e-02,-1.61357194e-01,3.85792315e-01,6.71366930e-01,
-8.76040459e-02,-3.02081332e-02,-7.85387680e-02,-2.87150204e-01,-3.62190865e-02,
-8.09790939e-02,-1.80272534e-01,-9.03454497e-02,-5.93765974e-02,-1.80922776e-01,
-3.34443092e-01,4.27656114e-01,2.77958333e-01,-3.65498275e-01,-4.44624603e-01,
-3.33298624e-01,3.37956846e-01,-9.99364723e-03,1.59281924e-01,-3.67539883e-01,
1.97324350e-01,-4.20186549e-01,-1.76442772e-01,3.49489361e-01,1.50009871e-01,
1.46227196e-01,-3.78889471e-01,1.01536829e-02,-8.07234719e-02,3.40434432e-01,
-2.97840685e-01,6.50868416e-02,3.86624694e-01,8.94862637e-02,1.46474093e-01,
-2.17987582e-01,5.78046478e-02,-1.87726878e-02,-5.98232299e-02,-3.19179624e-01,
-8.32483843e-02,7.65957907e-02,-1.05988890e-01,1.18702479e-01,-1.04717709e-01,
2.49496385e-01,-7.17857704e-02,1.51130691e-01,7.95488507e-02,1.89800903e-01,
4.03778031e-02,-1.99649110e-02,-4.63811681e-02,7.43079036e-02,-2.03792214e-01,
-1.37510508e-01,-5.30320639e-03,-1.30396232e-01,-1.44848570e-01,-1.08275615e-01,
-1.90553349e-02,1.02221541e-01,1.03002690e-01,1.36020929e-02,-1.39752135e-01,
-2.84113903e-02,1.58404246e-01,-1.19145878e-01,3.27202352e-03,-5.07250130e-02,
-1.30365312e-01,-4.26333919e-02,3.52876671e-02,1.32976323e-01,7.59094656e-02,
-6.38240762e-03,1.10654466e-01,7.96849281e-02,2.30799187e-02,1.51582118e-02,
-1.53049901e-01,-2.57429034e-01,-1.52225956e-01,6.60284162e-02,1.49973914e-01,
-1.35547549e-01,3.34135182e-02,5.52762076e-02,-3.78863364e-02,7.24897757e-02,
-1.03240728e-01,-1.35420114e-02,3.01801097e-02,-2.40372807e-01,-1.55299410e-01,
7.77367875e-02,-1.24001145e-01,-5.61420880e-02,-5.66705037e-03,-2.94228178e-02,
1.92717716e-01,5.67775853e-02,1.11508323e-02,1.02715209e-01,1.86036244e-01,
1.59761891e-01,-2.26212591e-02,-5.79044819e-02,-5.39625101e-02,-1.22253358e-01,
-6.17608353e-02,-3.49847116e-02,-1.57355741e-01,1.00760929e-01,4.20159884e-02,
-6.60344884e-02,5.27244546e-02,1.33613020e-01,-5.38502373e-02,1.38297528e-01,
3.36427167e-02,-3.25206853e-02,4.78228480e-02,1.51822507e-01,1.01110473e-01,
1.33769335e-02,-3.09242215e-03,-3.62791074e-03,1.50579810e-02,-5.41533008e-02,
1.10440522e-01,1.00492716e-01,7.78854564e-02,2.17735879e-02,-6.26308396e-02,
7.19641596e-02,2.14373022e-01,4.16182280e-02,-1.07728653e-01,2.60540754e-01,
-8.82555395e-02,1.62596598e-01,3.93707268e-02,-2.49247812e-02,4.70961491e-03,
1.71048328e-01,7.15696737e-02,-1.25860488e-02,6.16484992e-02,3.94553877e-03,
-7.78491050e-03,2.13099897e-01,-1.51443435e-02,2.46019110e-01,-1.94330260e-01,
-6.93377480e-02,1.62370741e-01,-1.64722785e-01,7.00592846e-02,5.17931990e-02,
-7.35005960e-02,4.08529416e-02,3.66773084e-02,1.04490347e-01,1.01373494e-01,
9.21172947e-02,2.00417694e-02,-1.64164349e-01,2.50968779e-03,-2.02906996e-01,
1.30768374e-01,1.57979473e-01,-1.00731682e-02,1.17010269e-02,-1.52695566e-01,
1.80813134e-01,-6.80533871e-02,5.30921221e-02,4.25355174e-02,1.16005689e-01,
3.19500314e-03,1.34659261e-02,6.88572675e-02,-1.40581280e-01,-1.15988426e-01,
3.36899422e-02,-7.44329393e-02,-8.45497549e-02,-5.80037758e-03,3.16857696e-02,
2.77275294e-02,-2.00868361e-02,-6.99692741e-02,4.99714282e-04,-8.32509249e-02,
-1.82667881e-01,7.79119581e-02,-1.11269690e-01,-1.31808132e-01,9.76681635e-02,
4.90867943e-02,-1.90561756e-01,2.16398865e-01,-1.88896611e-01,1.44654468e-01,
-8.13676864e-02,-1.21340439e-01,1.69398896e-02,1.08090118e-01,-1.64497942e-01,
6.04318678e-01,-1.90699920e-01,-2.99880326e-01,3.63137990e-01,5.04142761e-01,
1.51007161e-01,-4.48372006e-01,1.29417032e-01,-3.28397989e-01,9.21088517e-01,
-6.92636907e-01,2.19866559e-01,6.04490303e-02,1.17322639e-01,-2.43122667e-01,
-2.25074694e-01,6.82580054e-01,2.38671660e-01,1.57528967e-01,-6.47975564e-01,
-7.11128190e-02,8.30585659e-02,-2.74435699e-01,-4.69934978e-02,-1.15681119e-01,
-1.64055184e-01,3.92655768e-02,1.44214138e-01,9.01871026e-02,4.21197951e-01,
1.63558707e-01,5.43960929e-02,4.64137346e-02,2.39016071e-01,2.21892372e-01,
1.10977693e-02,1.68579787e-01,5.42099886e-02,1.16550788e-01,2.05354914e-01,
-5.20927250e-01,4.52886999e-01,5.29617555e-02,-5.87109268e-01,-4.53126371e-01,
-4.21585649e-01,6.85018420e-01,2.29944497e-01,7.50915825e-01,-8.27303469e-01,
-4.25806493e-02,-3.63566875e-01,-2.06865549e-01,5.92546046e-01,3.41280520e-01,
-3.98333110e-02,-5.20955145e-01,-2.26629257e-01,-2.41585553e-01,6.76254332e-01,
-1.46514043e-01,7.30411932e-02,4.05375570e-01,1.33451656e-01,-1.04780167e-01,
-1.28610745e-01,2.34845594e-01,6.95976242e-02,-3.11395768e-02,-4.33211494e-03,
1.81232199e-01,-1.38238311e-01,-1.30221182e-02,-1.45318434e-02,9.61262882e-02,
-1.22995868e-01,8.39923471e-02,6.19828552e-02,-2.97894944e-02,-4.45482582e-02,
-8.62067193e-02,-5.43029904e-02,-4.16467264e-02,-1.30074695e-01,1.03153944e-01,
1.76894367e-01,-5.73316589e-02,1.56106949e-01,-3.40049528e-02,5.55089675e-04,
1.59126401e-01,-4.60783206e-02,1.67752206e-02,-7.88622573e-02,-1.21971786e-01,
-1.01594448e-01,-4.14671302e-02,3.53154726e-02,-6.93688542e-02,2.13058472e-01,
1.97138973e-02,-2.66782884e-02,-3.01941596e-02,-1.74530268e-01,1.83517754e-01,
-2.05657065e-01,2.28307456e-01,-4.38887142e-02,-8.46125484e-02,4.56060246e-02,
-3.28919925e-02,-4.50686254e-02,1.63045987e-01,-1.60204187e-01,-7.60247931e-02,
1.19222246e-01,9.12308991e-02,-1.06583618e-01,-1.17446415e-01,-1.57878906e-01,
-8.77211392e-02,2.80488729e-02,1.06187657e-01,2.77292989e-02,-1.13624753e-02,
-7.97902327e-03,1.14371300e-01,1.13088958e-01,1.54198751e-01,2.41390571e-01,
2.34180048e-01,3.60784046e-02,3.06105375e-01,-7.93952271e-02,9.13393572e-02,
9.30454433e-02,1.24993585e-01,-8.65655094e-02,7.19468528e-03,4.25603017e-02,
3.12848426e-02,5.65861613e-02,-2.65260730e-02,-6.80348575e-02,-9.19707566e-02,
9.11240727e-02,4.23700958e-02,-2.32880767e-02,-1.28759965e-01,-1.30318567e-01,
1.56867877e-01,-4.99755368e-02,1.53620198e-01,6.49394542e-02,-7.82414675e-02,
-1.85940415e-01,9.23079476e-02,-1.84949376e-02,-5.55656989e-05,-5.84896915e-02,
9.56599414e-03,2.19954345e-02,2.16347277e-01,-3.11314985e-02,2.08770201e-01,
1.58884197e-01,1.72210559e-01,3.76543440e-02,4.94757034e-02,-1.88589469e-01,
1.31011531e-01,1.65513963e-01,5.53879291e-02,-4.42127436e-02,1.36750787e-01,
2.55172700e-01,8.16756114e-02,-4.26669344e-02,-7.43326033e-03,8.67159888e-02,
-6.72794729e-02,9.77837890e-02,-1.41216099e-01,-4.67757806e-02,4.70761582e-02,
-1.25349298e-01,2.40850493e-01,-1.88458227e-02,-4.75253984e-02,5.73125333e-02,
-5.97138703e-02,-3.25874425e-02,-1.05075121e-01,1.68972880e-01,1.11021586e-01,
-9.14278999e-02,5.00730798e-02,1.32234916e-01,6.16658740e-02,-1.71452448e-01,
7.95281529e-02,1.65248439e-01,-3.25032808e-02,1.14026345e-01,-9.46951509e-02,
1.75765604e-01,-6.56252056e-02,-1.62804291e-01,-2.76642591e-02,-1.04238406e-01,
6.57917634e-02,3.84893566e-02,-2.03246567e-02,-9.35739651e-02,3.84988673e-02,
-1.24566324e-01,1.39409438e-01,-6.71041980e-02,-8.98961630e-03,2.61104852e-01,
9.14405659e-02,1.21661713e-02,9.18285921e-02,1.42747583e-02,-3.73034440e-02,
4.71518487e-02,9.83461216e-02,1.68933883e-01,-1.02805323e-03,1.98786799e-02,
1.57938987e-01,-1.58339348e-02,4.84531000e-02,1.54280886e-01,5.67601584e-02,
-2.67745778e-02,-1.37785539e-01,-6.28660992e-03,1.11132987e-01,1.16483919e-01,
2.16238230e-01,-2.80816168e-01,-1.73826337e-01,5.80493510e-01,6.84650242e-01,
4.37299728e-01,-8.12318325e-01,9.00661498e-02,-7.96571910e-01,1.37597942e+00,
-2.42117196e-01,1.38612047e-01,1.91791803e-01,-3.24493170e-01,-1.51049852e-01,
-4.16065633e-01,6.20375216e-01,-9.56539437e-02,3.81025523e-01,-1.30203652e+00,
-1.79701038e-02,3.83959293e-01,2.22890496e-01,2.45772049e-01,2.78703868e-03,
-1.44103793e-02,8.09611976e-02,-5.30372025e-04,-1.15145221e-01,6.31621122e-01,
2.30470210e-01,-1.05699569e-01,7.33473524e-02,2.12321326e-01,6.73638657e-02,
-1.40594229e-01,1.31018698e-01,2.07240526e-02,3.64460200e-02,8.12375844e-02,
-1.32710814e-01,5.91575742e-01,-1.40631422e-01,-1.25787401e+00,-9.36630428e-01,
-9.41633046e-01,1.26760340e+00,5.04289288e-03,1.17496943e+00,-1.51439798e+00,
-8.41218904e-02,-2.71334082e-01,-1.35475859e-01,6.63006783e-01,2.89842695e-01,
4.02742714e-01,-7.10550308e-01,-1.77535877e-01,-7.67494261e-01,9.23803270e-01,
7.66203105e-02,1.19246438e-01,1.32906526e-01,-6.84400871e-02,8.98832008e-02,
-8.39993134e-02,3.21032703e-01,-8.45326632e-02,-1.24093421e-01,-2.58302271e-01,
2.12662295e-03,4.04840261e-02,-1.59636125e-01,-2.93258615e-02,-4.46097329e-02,
1.09581880e-01,-6.32943660e-02,1.91112354e-01,-5.67296185e-02,1.48206025e-01,
}; 
k2c_tensor conv1d_4_kernel = {&conv1d_4_kernel_array[0],3,960,{ 4,24,10, 1, 1}}; 
float conv1d_4_bias_array[10] = {
-1.41734944e-03,-1.49785923e-02,-1.37653062e-02,-1.42465821e-02,1.34020916e-03,
1.32671604e-02,-3.61614414e-02,-3.78044369e-03,-1.59429237e-02,-1.04229441e-02,
}; 
k2c_tensor conv1d_4_bias = {&conv1d_4_bias_array[0],1,10,{10, 1, 1, 1, 1}}; 

 
size_t conv1d_9_stride = 1; 
size_t conv1d_9_dilation = 1; 
float conv1d_9_output_array[330] = {0}; 
k2c_tensor conv1d_9_output = {&conv1d_9_output_array[0],2,330,{33,10, 1, 1, 1}}; 
float conv1d_9_padded_input_array[864] = {0}; 
k2c_tensor conv1d_9_padded_input = {&conv1d_9_padded_input_array[0],2,864,{36,24, 1, 1, 1}}; 
size_t conv1d_9_pad[2] = {1,2}; 
float conv1d_9_fill = 0.0f; 
float conv1d_9_kernel_array[960] = {
-1.02214880e-01,-2.27591917e-01,1.16355971e-01,2.37040613e-02,-4.96315397e-02,
4.51703966e-02,3.14130574e-01,-2.74340153e-01,1.36318495e-02,-2.23034576e-01,
5.00838272e-03,1.10564783e-01,1.63534451e-02,6.06235266e-02,3.13464105e-02,
-9.33907777e-02,3.38209495e-02,-1.16113581e-01,1.24651007e-02,1.25239998e-01,
-3.37820314e-02,-4.47495542e-02,-1.39671579e-01,1.63598061e-01,-7.25769475e-02,
-3.92831303e-02,1.24028578e-01,-3.50043885e-02,2.10307408e-02,-2.80031692e-02,
-1.49796769e-01,-6.56264722e-02,-2.13747397e-01,6.90942705e-02,1.59745798e-01,
4.46244031e-02,4.34942581e-02,-2.49161553e-02,-2.99882609e-02,1.24686472e-01,
4.11181785e-02,2.43218973e-01,1.02198631e-01,-1.78541020e-01,-6.76469728e-02,
6.39154911e-02,-1.86098814e-01,2.79193163e-01,-1.06915228e-01,1.12955019e-01,
-5.79401366e-02,-5.71781248e-02,7.46300966e-02,-1.73185617e-02,-9.71248746e-03,
2.71334164e-02,3.27553228e-02,-3.24393250e-02,4.99848090e-03,-1.36451706e-01,
5.57574704e-02,4.79189269e-02,-4.05055434e-02,4.86363471e-02,-5.69119826e-02,
5.40315472e-02,1.81309953e-02,6.97629675e-02,6.15681261e-02,1.17718436e-01,
-4.74755019e-02,-7.62815624e-02,1.17701299e-01,-4.79949340e-02,4.69600782e-03,
4.64010332e-03,1.07529514e-01,-1.44712418e-01,6.22703396e-02,-4.52981554e-02,
2.95880865e-02,-1.15072630e-01,-1.99847799e-02,-3.28062139e-02,-8.76538903e-02,
-7.64158145e-02,1.26737073e-01,8.09231177e-02,-1.31586222e-02,6.84661493e-02,
-1.76458403e-01,-8.39616284e-02,-3.58241528e-01,-1.62531823e-01,-1.17699601e-01,
1.09067231e-01,-1.57720178e-01,-1.33781701e-01,-1.60515264e-01,-5.27949035e-01,
-1.51476532e-01,1.41334176e-01,1.30577326e-01,-9.59533732e-03,9.40043628e-02,
1.12121776e-01,-4.89162914e-02,-2.01964065e-01,-8.11380148e-02,-1.91849589e-01,
3.32997218e-02,8.04145262e-02,3.85392718e-02,9.16689709e-02,2.18182966e-01,
2.07884703e-02,-2.15323582e-01,1.21481560e-01,-9.80072394e-02,1.18558653e-01,
1.05332904e-01,8.30725655e-02,1.69041127e-01,1.41343223e-02,2.28676319e-01,
1.85116321e-01,-2.66779542e-01,-2.34969817e-02,-2.57694591e-02,1.89747922e-02,
6.52832910e-02,8.09258446e-02,2.94446409e-01,-1.44801989e-01,-1.95353776e-02,
-3.37768674e-01,-2.12215438e-01,-4.64630947e-02,9.23278108e-02,-3.33297879e-01,
-7.67917112e-02,-2.77452707e-01,-4.81334291e-02,4.41143923e-02,-7.05128610e-02,
-1.49666324e-01,-4.16259952e-02,1.02845564e-01,1.95281580e-01,-2.35902578e-01,
-1.52305782e-01,2.07205698e-01,4.88625541e-02,-1.30877048e-01,-3.36098760e-01,
-1.66638494e-01,2.07547247e-01,-8.73525962e-02,1.64441951e-02,-4.95946035e-02,
-1.83740571e-01,-3.37390691e-01,6.63773000e-01,-5.05144060e-01,3.95530909e-01,
-6.67376816e-02,2.01180413e-01,-4.37425911e-01,-5.09591326e-02,1.63728576e-02,
1.74151704e-01,-3.27401966e-01,1.63377857e+00,5.65726280e-01,-2.17906237e-01,
3.85238260e-01,-4.75692987e-01,6.91034198e-02,-2.47502830e-02,1.63608879e-01,
5.53444028e-01,-4.92044270e-01,2.45781168e-01,2.28986531e-01,2.19410658e-01,
4.94328290e-02,1.81406125e-01,4.33359444e-02,-1.11199498e-01,1.01315968e-01,
8.34810495e-01,3.67982060e-01,1.00725305e+00,7.39683270e-01,5.32622099e-01,
2.30006859e-01,7.77520537e-01,7.45856404e-01,2.04010308e-01,-6.30765501e-03,
4.50604409e-02,2.40585487e-03,3.93847637e-02,-1.23682491e-01,-8.49068761e-02,
-1.68361008e-01,-4.40212945e-03,6.84118271e-02,7.93546438e-02,1.35247096e-01,
-2.50827670e-01,-2.28433698e-01,1.13472509e+00,-6.07858360e-01,-5.51344275e-01,
-5.52402698e-02,-7.45937765e-01,-5.89644551e-01,-3.53784144e-01,5.34675503e-03,
-1.95901886e-01,1.68771461e-01,1.19932376e-01,7.68287405e-02,2.56764948e-01,
1.77075729e-01,1.63397454e-02,-1.13651846e-02,-3.19864333e-01,2.01385900e-01,
-6.13943160e-01,1.50505424e+00,-3.41542721e-01,-1.46420598e+00,-4.38292734e-02,
-5.79444051e-01,-2.58481532e-01,7.52053797e-01,5.75589359e-01,1.26802787e-01,
2.09114343e-01,5.80149963e-02,6.85723200e-02,-1.87793463e-01,-1.09318726e-01,
1.30714342e-01,-4.37552854e-02,1.38943255e-01,-1.99953970e-02,2.23327763e-02,
-5.33361174e-02,-4.15201597e-02,-3.45366597e-02,4.16839914e-03,1.11842535e-01,
1.38694033e-01,7.88888037e-02,-5.24412021e-02,-7.23906606e-02,-7.97093213e-02,
1.48311242e-01,-1.81269377e-01,-3.56518663e-02,1.91084936e-01,3.78624648e-02,
5.75853453e-04,2.53089648e-02,1.51963066e-02,1.32091105e-01,6.76836586e-04,
-1.81358941e-02,-3.97989638e-02,6.00532070e-03,9.02315825e-02,2.28432983e-01,
1.48608074e-01,-1.03395298e-01,-1.26393110e-01,-2.58230805e-01,-1.25744328e-01,
-4.54525091e-02,-9.04259980e-02,9.99599993e-02,4.22525182e-02,1.05758160e-01,
-1.11809276e-01,1.47957969e-02,-5.91257997e-02,4.99279164e-02,5.46659939e-02,
3.66268195e-02,5.99967539e-02,-2.03449675e-03,9.61736962e-02,-2.48387177e-02,
1.50376167e-02,1.18844137e-01,1.23239771e-01,-7.69601539e-02,9.01009291e-02,
-5.11242524e-02,-1.60058029e-02,1.82615314e-02,2.06763353e-02,-6.44625351e-02,
3.19482833e-02,4.82774433e-03,-1.60283282e-01,1.23410918e-01,-4.79501188e-02,
2.98624691e-02,-4.66582738e-02,8.82092416e-02,-6.55548200e-02,-2.48286240e-02,
8.82888064e-02,5.07439300e-02,3.08849253e-02,4.47218604e-02,-5.53165078e-02,
3.19594480e-02,-2.31525358e-02,-4.37436625e-02,1.16059473e-02,1.95125625e-01,
-7.43938377e-03,-2.44685307e-01,2.10200138e-02,-5.49713001e-02,2.41238941e-02,
-1.52359754e-01,5.23822963e-01,4.59981225e-02,1.09898001e-01,-3.69901538e-01,
-8.93670544e-02,-2.38428622e-01,-1.97655588e-01,2.58554667e-01,-4.05948646e-02,
-1.26702618e-02,2.59240139e-02,-6.98095141e-03,-5.22642024e-02,9.07029882e-02,
5.78942522e-02,-5.48750013e-02,2.93814600e-01,8.72186571e-03,2.09533051e-02,
-1.23143736e-02,-1.52239231e-02,-1.11658275e-01,2.95568537e-02,-2.56663352e-01,
-5.03786653e-02,4.65151854e-02,4.89160009e-02,1.04384005e-01,-2.08575159e-01,
-1.83691964e-01,-2.68460810e-02,1.73036128e-01,-1.09432250e-01,-2.44204085e-02,
-3.23170312e-02,-2.39330269e-02,-3.87135446e-02,8.59306753e-02,3.43896225e-02,
1.33377820e-01,-4.67933118e-02,1.19278304e-01,-2.99435914e-01,5.18154353e-02,
2.48434857e-01,1.12358876e-01,3.14690441e-01,1.72481254e-01,-2.33088464e-01,
-9.39455405e-02,-7.10908696e-02,1.16391338e-01,-2.15533301e-01,1.84338212e-01,
3.33380746e-03,-2.33042687e-01,6.53333664e-02,1.36758670e-01,-1.11445114e-01,
3.29146124e-02,-1.30810170e-03,-1.26737684e-01,9.46970806e-02,5.21210767e-02,
-5.07649630e-02,9.93010029e-02,2.14796633e-01,-6.56512827e-02,-1.84028894e-02,
1.20451212e-01,1.49244964e-01,3.51648360e-01,-5.77188253e-01,3.45737427e-01,
1.94540083e-01,-1.11255206e-01,-2.36276552e-01,-5.26268687e-03,1.73830897e-01,
6.61026463e-02,-1.83482677e-01,1.42437208e+00,8.99052680e-01,-3.15017760e-01,
6.16154015e-01,-2.49154046e-01,2.70164311e-01,1.15493022e-01,4.88443196e-01,
-1.67855188e-01,2.18513027e-01,-1.19302258e-01,5.65061644e-02,-2.32515894e-02,
1.22153789e-01,-4.34747458e-01,1.23194546e-01,8.74072090e-02,-4.93999124e-02,
4.29609686e-01,4.46147233e-01,8.41628671e-01,5.88577211e-01,7.44146526e-01,
3.15776467e-01,1.24217474e+00,8.58792961e-01,6.31610870e-01,-1.57605693e-01,
5.12949638e-02,-1.19207799e-01,-3.31936739e-02,-5.81430271e-02,-5.45189753e-02,
-5.94558753e-02,-7.36241564e-02,3.30329426e-02,5.29504381e-02,-2.14633390e-01,
-2.21635088e-01,-2.57650346e-01,1.22738469e+00,-4.48913872e-01,-1.86520800e-01,
7.87760094e-02,-7.36721158e-01,-2.15701446e-01,8.77635612e-04,1.54389530e-01,
2.54514456e-01,4.00216356e-02,5.07663846e-01,3.07637751e-02,-7.19926208e-02,
9.22820941e-02,-1.73324987e-01,-1.46350488e-01,1.45390660e-01,-2.25999519e-01,
2.44218484e-02,1.51803136e-01,-1.80667654e-01,-1.38115978e+00,7.01804221e-01,
-5.32294512e-01,1.02141833e+00,-7.81103969e-02,1.03742089e-02,-4.43216652e-01,
-2.11285632e-02,1.59945041e-01,-1.48965597e-01,4.97793639e-03,-2.09833577e-01,
-1.79646611e-01,8.30323175e-02,5.86524308e-02,-2.04755533e-02,2.05762982e-01,
1.77926645e-01,1.58378169e-01,2.26598270e-02,-7.55057996e-03,-1.10930048e-01,
-1.44879445e-02,5.87598011e-02,1.34835821e-02,1.38641410e-02,2.69292235e-01,
5.27014434e-02,2.05294326e-01,-1.71422660e-01,6.46499321e-02,-2.64177382e-01,
-1.24293782e-01,2.80237105e-02,2.94107962e-02,-7.00672995e-03,7.06282258e-02,
2.37165876e-02,-1.26079157e-01,1.29506961e-02,4.74640280e-02,-6.11486807e-02,
1.28047466e-01,1.56245559e-01,-7.82049000e-02,4.04807879e-03,-1.13137029e-01,
-9.01324004e-02,-3.34850587e-02,-3.61894220e-02,-1.56520620e-01,-7.70937419e-03,
1.00417376e-01,-7.62244165e-02,8.30574632e-02,-1.99848339e-02,-1.02289297e-01,
1.32694915e-02,-7.50325918e-02,-9.26569998e-02,-3.32308970e-02,-2.24373606e-03,
3.56368418e-03,-6.21786118e-02,1.60754576e-01,6.65237084e-02,7.95854777e-02,
5.81552163e-02,5.32390177e-02,-1.57689020e-01,8.69936273e-02,-5.29931635e-02,
2.30100993e-02,-8.04006010e-02,6.59844130e-02,4.81695719e-02,9.71626490e-03,
-2.92616989e-02,-3.68596800e-02,3.52843478e-02,-9.99629423e-02,-9.58869886e-03,
2.76569761e-02,1.00353874e-01,1.13993417e-02,-6.26638830e-02,9.22072455e-02,
1.47201503e-02,2.45825127e-02,7.07148165e-02,-1.41131878e-01,-8.73690173e-02,
-1.33710235e-01,-2.16286983e-02,-2.09885947e-02,2.78227665e-02,8.26324522e-03,
-8.60522464e-02,3.60008813e-02,2.54261140e-02,3.90078276e-01,2.81372160e-01,
-6.47739321e-02,-7.81612545e-02,-3.43322486e-01,3.03881653e-02,-1.94221526e-01,
-7.34696090e-02,2.92752981e-02,2.18047157e-01,-2.01019078e-01,-6.28827140e-02,
-1.51282579e-01,-1.67800821e-02,3.62662934e-02,-5.53299636e-02,2.80987862e-02,
-2.02312917e-01,3.93155180e-02,1.58891473e-02,3.41718420e-02,2.04031598e-02,
1.17502157e-02,8.15743431e-02,8.92511532e-02,-1.32529095e-01,-1.22161889e-02,
-5.88553138e-02,1.10412344e-01,1.00124799e-01,-4.33990406e-03,9.65367258e-02,
-4.84673940e-02,-2.56086402e-02,-2.00374201e-01,-1.35914028e-01,1.84405111e-02,
1.07958108e-01,-9.54479426e-02,-3.02325964e-01,-1.84807345e-01,-4.75984812e-02,
-4.11504358e-02,-4.09099340e-01,2.69445777e-01,1.99663728e-01,-2.71845341e-01,
-6.67724162e-02,1.25703827e-01,2.27299109e-01,-1.42347002e-02,-4.42158990e-02,
4.93532307e-02,-2.13772044e-01,-7.21540004e-02,-4.82943170e-02,-3.25604349e-01,
2.11547747e-01,1.62738357e-02,-1.40590087e-01,-6.60829023e-02,-5.89892827e-02,
-1.05934106e-01,-1.53605968e-01,1.81461126e-01,-3.79021242e-02,4.22015712e-02,
-2.66636517e-02,-9.58183706e-02,2.95036733e-01,-4.80984479e-01,1.91115797e-01,
1.09371796e-01,1.64255664e-01,-2.46844813e-01,-1.62826896e-01,1.24736637e-01,
7.10503533e-02,-1.85793146e-01,1.48621893e+00,9.34507132e-01,-2.77888030e-01,
8.82437348e-01,-6.11983538e-01,5.62774241e-02,-2.70019859e-01,2.47751355e-01,
4.54683192e-02,2.59287477e-01,4.74733599e-02,-8.75154510e-02,8.16722065e-02,
1.38047993e-01,4.35070135e-02,1.61783710e-01,1.26911461e-01,3.98046553e-01,
1.73130915e-01,1.36216596e-01,4.08158690e-01,6.66234136e-01,1.33904189e-01,
2.15980522e-02,4.83453184e-01,5.67904532e-01,6.38746262e-01,-3.20266071e-03,
4.43064887e-03,-2.33854409e-02,-3.08440533e-02,-7.27527440e-02,-8.25876817e-02,
-1.02017343e-01,5.01485914e-02,1.11516610e-01,5.51589541e-02,4.66961674e-02,
-5.54384068e-02,-2.50578448e-02,1.44039047e+00,-2.43683413e-01,1.45440409e-02,
-4.70962971e-02,-7.32043326e-01,-1.57323331e-01,-1.34744048e-01,-2.07256293e-03,
9.39374566e-02,3.64985056e-02,2.20212355e-01,4.63307612e-02,1.15004092e-01,
-2.61780918e-01,-1.17603436e-01,5.54371141e-02,2.72291303e-01,6.75095841e-02,
1.79194927e-01,-2.29068905e-01,-2.87535995e-01,-1.34504628e+00,-5.82386833e-03,
-2.04753771e-01,3.67501259e-01,-3.14627498e-01,5.24619222e-02,-3.52378011e-01,
7.05066253e-04,-3.67258549e-01,1.19771108e-01,-9.11991205e-03,1.56161129e-01,
1.19716279e-01,-2.06417441e-02,4.26204316e-02,-5.50999232e-02,-3.66224974e-01,
2.05012932e-02,-1.72004551e-01,-1.13248676e-01,7.44104087e-02,6.15702085e-02,
-1.19621135e-01,-4.46327068e-02,1.15703139e-02,1.85210869e-01,3.35445106e-02,
1.69895917e-01,-2.04759419e-01,-1.28251016e-01,1.45816058e-01,1.60439424e-02,
-1.51677206e-02,3.36651579e-02,1.97042692e-02,1.10632703e-01,-1.51435155e-02,
-1.07052952e-01,2.38692805e-01,-9.78216305e-02,1.23142675e-01,-6.59362227e-02,
5.46056628e-02,-1.06333606e-02,-6.91183209e-02,-3.39344777e-02,-4.88878265e-02,
-5.66198006e-02,2.22257942e-01,5.88267855e-02,-1.37870274e-02,1.31560385e-01,
8.23375285e-02,6.03358597e-02,-1.93396315e-01,-8.77625942e-02,2.26545572e-01,
-1.30097091e-01,3.01757138e-02,-1.91018134e-01,1.87963605e-01,-5.29865623e-02,
7.51407966e-02,7.45766237e-02,4.49929424e-02,-2.21251436e-02,-6.74706846e-02,
2.33819501e-05,3.57152894e-02,-2.69716009e-02,1.31652709e-02,1.40841797e-01,
1.00776568e-01,-1.50943846e-01,1.53596744e-01,-1.07734345e-01,6.21341541e-03,
6.49833679e-03,4.29264642e-02,-1.12500206e-01,2.60017663e-02,1.14991061e-01,
1.41914144e-01,-5.53881004e-02,5.82354851e-02,-2.09688414e-02,1.59110129e-01,
9.09629241e-02,1.48488447e-01,-3.47350258e-03,1.99955493e-01,-9.44107920e-02,
-1.29898220e-01,3.06398906e-02,-2.64996111e-01,8.97814184e-02,-9.58878249e-02,
-1.67382240e-01,-2.33733490e-01,8.20463151e-02,1.99414939e-01,-1.90171212e-01,
6.09081648e-02,-1.66113436e-01,-4.76860218e-02,-1.28000885e-01,3.78378592e-02,
1.18764430e-01,-9.96823013e-02,6.58569336e-02,-1.02096610e-02,-7.28940293e-02,
-1.64005160e-02,-1.22104801e-01,1.59668867e-02,8.42679441e-02,-2.02934772e-01,
1.33868814e-01,2.80526698e-01,7.06254318e-02,-1.29982606e-02,-3.03740501e-01,
-1.56855136e-01,1.22154765e-01,-1.31069660e-01,9.44870114e-02,2.07029819e-01,
-4.97974455e-02,-4.63370085e-02,-2.66199727e-02,2.42584705e-01,-1.09382585e-01,
1.18600115e-01,3.73290926e-01,-2.61443973e-01,8.62637535e-03,4.66607586e-02,
2.46549740e-01,1.73138797e-01,-1.00424320e-01,-4.24340516e-01,-1.37760386e-01,
1.69491485e-01,-3.96266252e-01,3.19838256e-01,-9.29118171e-02,-1.48918226e-01,
-6.57002348e-03,1.94323197e-01,1.20280184e-01,-2.03393269e-02,3.19802724e-02,
-1.47252589e-01,-3.97208780e-02,3.34405452e-02,-4.90007065e-02,2.22374409e-01,
1.87470138e-01,-5.81259094e-02,8.00444037e-02,-1.50178269e-01,1.57154649e-01,
4.08647731e-02,-1.49288192e-01,2.50751317e-01,-2.10774049e-01,9.49570164e-02,
-8.10241401e-02,-2.15067446e-01,6.11902714e-01,-6.06985748e-01,6.79020062e-02,
2.38667011e-01,3.42525363e-01,-2.91053236e-01,-2.10018665e-01,2.33185589e-02,
-1.48332968e-01,1.80438384e-01,2.14281416e+00,1.87133241e+00,-4.71262425e-01,
8.63109052e-01,-1.20897658e-01,4.40681875e-01,3.54979247e-01,4.40865934e-01,
-3.50081623e-01,1.73868820e-01,3.88922021e-02,-5.22954881e-01,-1.85696315e-02,
1.35698050e-01,2.33464539e-02,-1.18102647e-01,1.99502990e-01,1.80092528e-01,
4.30320889e-01,9.87623632e-02,1.03990033e-01,4.29770201e-01,-2.98787951e-01,
4.54897165e-01,-6.17806256e-01,3.98567796e-01,6.39445543e-01,6.82076961e-02,
9.39811580e-03,-5.76989055e-02,2.13733137e-01,-2.07336724e-01,5.77666890e-03,
-1.03440024e-02,-2.09864359e-02,-5.28449267e-02,-4.82088514e-03,-2.26718694e-01,
-1.52512625e-01,7.91812688e-02,1.68981421e+00,1.83077127e-01,-1.12413382e-02,
1.32861570e-01,-4.24333662e-01,-1.84321895e-01,-2.78266400e-01,-2.75046770e-02,
7.34967813e-02,1.47636309e-01,6.99410811e-02,-5.25329374e-02,9.84132476e-03,
2.53066514e-02,-3.45734060e-02,2.72284299e-01,-4.81254756e-02,1.10180058e-01,
5.76366365e-01,-1.59157544e-01,-1.00657809e+00,-2.02036572e+00,7.52204955e-02,
-1.63334697e-01,5.74186221e-02,1.31621107e-01,2.73774445e-01,-9.73882526e-02,
}; 
k2c_tensor conv1d_9_kernel = {&conv1d_9_kernel_array[0],3,960,{ 4,24,10, 1, 1}}; 
float conv1d_9_bias_array[10] = {
1.67865795e-03,5.38741099e-03,4.43378948e-02,-1.86709082e-03,-6.57611294e-03,
-2.20143665e-02,1.21485218e-02,-2.46940665e-02,-1.82856154e-02,1.28324851e-02,
}; 
k2c_tensor conv1d_9_bias = {&conv1d_9_bias_array[0],1,10,{10, 1, 1, 1, 1}}; 

 
size_t conv1d_5_stride = 1; 
size_t conv1d_5_dilation = 1; 
float conv1d_5_output_array[33] = {0}; 
k2c_tensor conv1d_5_output = {&conv1d_5_output_array[0],2,33,{33, 1, 1, 1, 1}}; 
float conv1d_5_padded_input_array[360] = {0}; 
k2c_tensor conv1d_5_padded_input = {&conv1d_5_padded_input_array[0],2,360,{36,10, 1, 1, 1}}; 
size_t conv1d_5_pad[2] = {1,2}; 
float conv1d_5_fill = 0.0f; 
float conv1d_5_kernel_array[40] = {
-9.81634855e-02,2.86442995e-01,1.60519183e-01,1.06422352e-02,8.28288570e-02,
-1.49905831e-01,-1.67944178e-01,1.09298579e-01,-1.71559036e-01,-5.01288660e-03,
-1.00951493e-01,-1.56646684e-01,1.18636017e-04,2.81192791e-02,-1.36217669e-01,
1.49910882e-01,1.30183473e-01,-1.59352541e-01,1.00923963e-01,-5.78446500e-02,
5.15247732e-02,-5.52084669e-02,3.07690911e-02,-9.67143625e-02,5.87077998e-03,
-2.99402457e-02,8.43861997e-02,1.24850139e-01,-1.32842883e-01,-1.56808034e-01,
9.07214433e-02,9.92389843e-02,-7.04204440e-02,-1.88342363e-01,-4.68334593e-02,
-4.81706597e-02,-3.62628736e-02,3.57693098e-02,2.51997352e-01,-9.10095796e-02,
}; 
k2c_tensor conv1d_5_kernel = {&conv1d_5_kernel_array[0],3,40,{ 4,10, 1, 1, 1}}; 
float conv1d_5_bias_array[1] = {
3.10088275e-04,}; 
k2c_tensor conv1d_5_bias = {&conv1d_5_bias_array[0],1,1,{1,1,1,1,1}}; 

 
size_t conv1d_10_stride = 1; 
size_t conv1d_10_dilation = 1; 
float conv1d_10_output_array[33] = {0}; 
k2c_tensor conv1d_10_output = {&conv1d_10_output_array[0],2,33,{33, 1, 1, 1, 1}}; 
float conv1d_10_padded_input_array[360] = {0}; 
k2c_tensor conv1d_10_padded_input = {&conv1d_10_padded_input_array[0],2,360,{36,10, 1, 1, 1}}; 
size_t conv1d_10_pad[2] = {1,2}; 
float conv1d_10_fill = 0.0f; 
float conv1d_10_kernel_array[40] = {
1.11807324e-01,1.71017244e-01,-9.03583616e-02,-1.19838186e-01,2.08320007e-01,
-4.97861616e-02,2.32249498e-01,1.11808941e-01,1.17386952e-01,-1.44809827e-01,
1.25202900e-02,-1.41521797e-01,-1.10681146e-01,-5.57293706e-02,-1.29463166e-01,
1.51257545e-01,-7.16025606e-02,-2.05310378e-02,-4.37963642e-02,1.46280766e-01,
-1.53945416e-01,3.40658762e-02,-2.75845099e-02,-7.08962381e-02,-2.91468743e-02,
-7.40814656e-02,-7.17506334e-02,5.21132015e-02,1.25691354e-01,-7.00582005e-03,
3.20734493e-02,-5.87455370e-02,-2.43473694e-01,-1.57405943e-01,5.82411012e-04,
-1.03588521e-01,-4.89299819e-02,-1.40370652e-01,-2.07718283e-01,-3.16746794e-02,
}; 
k2c_tensor conv1d_10_kernel = {&conv1d_10_kernel_array[0],3,40,{ 4,10, 1, 1, 1}}; 
float conv1d_10_bias_array[1] = {
8.41875374e-03,}; 
k2c_tensor conv1d_10_bias = {&conv1d_10_bias_array[0],1,1,{1,1,1,1,1}}; 

 
size_t target_temp_newndim = 1; 
size_t target_temp_newshp[K2C_MAX_NDIM] = {33, 1, 1, 1, 1}; 


size_t target_dens_newndim = 1; 
size_t target_dens_newshp[K2C_MAX_NDIM] = {33, 1, 1, 1, 1}; 


k2c_reshape(input_past_pinj_input,reshape_5_newshp,reshape_5_newndim); 
k2c_tensor reshape_5_output; 
reshape_5_output.ndim = input_past_pinj_input->ndim; // copy data into output struct 
reshape_5_output.numel = input_past_pinj_input->numel; 
memcpy(reshape_5_output.shape,input_past_pinj_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_5_output.array = &input_past_pinj_input->array[0]; // rename for clarity 
k2c_reshape(input_past_curr_input,reshape_7_newshp,reshape_7_newndim); 
k2c_tensor reshape_7_output; 
reshape_7_output.ndim = input_past_curr_input->ndim; // copy data into output struct 
reshape_7_output.numel = input_past_curr_input->numel; 
memcpy(reshape_7_output.shape,input_past_curr_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_7_output.array = &input_past_curr_input->array[0]; // rename for clarity 
k2c_reshape(input_past_tinj_input,reshape_9_newshp,reshape_9_newndim); 
k2c_tensor reshape_9_output; 
reshape_9_output.ndim = input_past_tinj_input->ndim; // copy data into output struct 
reshape_9_output.numel = input_past_tinj_input->numel; 
memcpy(reshape_9_output.shape,input_past_tinj_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_9_output.array = &input_past_tinj_input->array[0]; // rename for clarity 
k2c_reshape(input_past_gasA_input,reshape_11_newshp,reshape_11_newndim); 
k2c_tensor reshape_11_output; 
reshape_11_output.ndim = input_past_gasA_input->ndim; // copy data into output struct 
reshape_11_output.numel = input_past_gasA_input->numel; 
memcpy(reshape_11_output.shape,input_past_gasA_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_11_output.array = &input_past_gasA_input->array[0]; // rename for clarity 
k2c_reshape(input_future_pinj_input,reshape_4_newshp,reshape_4_newndim); 
k2c_tensor reshape_4_output; 
reshape_4_output.ndim = input_future_pinj_input->ndim; // copy data into output struct 
reshape_4_output.numel = input_future_pinj_input->numel; 
memcpy(reshape_4_output.shape,input_future_pinj_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_4_output.array = &input_future_pinj_input->array[0]; // rename for clarity 
k2c_reshape(input_future_curr_input,reshape_6_newshp,reshape_6_newndim); 
k2c_tensor reshape_6_output; 
reshape_6_output.ndim = input_future_curr_input->ndim; // copy data into output struct 
reshape_6_output.numel = input_future_curr_input->numel; 
memcpy(reshape_6_output.shape,input_future_curr_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_6_output.array = &input_future_curr_input->array[0]; // rename for clarity 
k2c_reshape(input_future_tinj_input,reshape_8_newshp,reshape_8_newndim); 
k2c_tensor reshape_8_output; 
reshape_8_output.ndim = input_future_tinj_input->ndim; // copy data into output struct 
reshape_8_output.numel = input_future_tinj_input->numel; 
memcpy(reshape_8_output.shape,input_future_tinj_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_8_output.array = &input_future_tinj_input->array[0]; // rename for clarity 
k2c_reshape(input_future_gasA_input,reshape_10_newshp,reshape_10_newndim); 
k2c_tensor reshape_10_output; 
reshape_10_output.ndim = input_future_gasA_input->ndim; // copy data into output struct 
reshape_10_output.numel = input_future_gasA_input->numel; 
memcpy(reshape_10_output.shape,input_future_gasA_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_10_output.array = &input_future_gasA_input->array[0]; // rename for clarity 
k2c_reshape(input_thomson_temp_EFITRT1_input,reshape_1_newshp,reshape_1_newndim); 
k2c_tensor reshape_1_output; 
reshape_1_output.ndim = input_thomson_temp_EFITRT1_input->ndim; // copy data into output struct 
reshape_1_output.numel = input_thomson_temp_EFITRT1_input->numel; 
memcpy(reshape_1_output.shape,input_thomson_temp_EFITRT1_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_1_output.array = &input_thomson_temp_EFITRT1_input->array[0]; // rename for clarity 
k2c_reshape(input_thomson_dens_EFITRT1_input,reshape_2_newshp,reshape_2_newndim); 
k2c_tensor reshape_2_output; 
reshape_2_output.ndim = input_thomson_dens_EFITRT1_input->ndim; // copy data into output struct 
reshape_2_output.numel = input_thomson_dens_EFITRT1_input->numel; 
memcpy(reshape_2_output.shape,input_thomson_dens_EFITRT1_input->shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_2_output.array = &input_thomson_dens_EFITRT1_input->array[0]; // rename for clarity 
k2c_concatenate(&concatenate_3_output,concatenate_3_axis,concatenate_3_num_tensors0,&reshape_5_output,&reshape_7_output,&reshape_9_output,&reshape_11_output); 
k2c_concatenate(&concatenate_2_output,concatenate_2_axis,concatenate_2_num_tensors0,&reshape_4_output,&reshape_6_output,&reshape_8_output,&reshape_10_output); 
k2c_concatenate(&concatenate_1_output,concatenate_1_axis,concatenate_1_num_tensors0,&reshape_1_output,&reshape_2_output); 
k2c_lstm(&lstm_1_output,&concatenate_3_output,lstm_1_state,&lstm_1_kernel, 
	&lstm_1_recurrent_kernel,&lstm_1_bias,lstm_1_fwork, 
	lstm_1_go_backwards,lstm_1_return_sequences, 
	k2c_hard_sigmoid,k2c_relu); 
k2c_lstm(&lstm_2_output,&concatenate_2_output,lstm_2_state,&lstm_2_kernel, 
	&lstm_2_recurrent_kernel,&lstm_2_bias,lstm_2_fwork, 
	lstm_2_go_backwards,lstm_2_return_sequences, 
	k2c_hard_sigmoid,k2c_relu); 
k2c_reshape(&concatenate_1_output,reshape_3_newshp,reshape_3_newndim); 
k2c_tensor reshape_3_output; 
reshape_3_output.ndim = concatenate_1_output.ndim; // copy data into output struct 
reshape_3_output.numel = concatenate_1_output.numel; 
memcpy(reshape_3_output.shape,concatenate_1_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_3_output.array = &concatenate_1_output.array[0]; // rename for clarity 
k2c_reshape(&lstm_1_output,reshape_12_newshp,reshape_12_newndim); 
k2c_tensor reshape_12_output; 
reshape_12_output.ndim = lstm_1_output.ndim; // copy data into output struct 
reshape_12_output.numel = lstm_1_output.numel; 
memcpy(reshape_12_output.shape,lstm_1_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_12_output.array = &lstm_1_output.array[0]; // rename for clarity 
k2c_reshape(&lstm_2_output,reshape_13_newshp,reshape_13_newndim); 
k2c_tensor reshape_13_output; 
reshape_13_output.ndim = lstm_2_output.ndim; // copy data into output struct 
reshape_13_output.numel = lstm_2_output.numel; 
memcpy(reshape_13_output.shape,lstm_2_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
reshape_13_output.array = &lstm_2_output.array[0]; // rename for clarity 
k2c_concatenate(&concatenate_4_output,concatenate_4_axis,concatenate_4_num_tensors0,&reshape_3_output,&reshape_12_output,&reshape_13_output); 
k2c_pad1d(&conv1d_1_padded_input,&concatenate_4_output,conv1d_1_fill, 
	conv1d_1_pad); 
k2c_conv1d(&conv1d_1_output,&conv1d_1_padded_input,&conv1d_1_kernel, 
	&conv1d_1_bias,conv1d_1_stride,conv1d_1_dilation,k2c_relu); 
k2c_pad1d(&conv1d_6_padded_input,&concatenate_4_output,conv1d_6_fill, 
	conv1d_6_pad); 
k2c_conv1d(&conv1d_6_output,&conv1d_6_padded_input,&conv1d_6_kernel, 
	&conv1d_6_bias,conv1d_6_stride,conv1d_6_dilation,k2c_relu); 
k2c_pad1d(&conv1d_2_padded_input,&conv1d_1_output,conv1d_2_fill, 
	conv1d_2_pad); 
k2c_conv1d(&conv1d_2_output,&conv1d_2_padded_input,&conv1d_2_kernel, 
	&conv1d_2_bias,conv1d_2_stride,conv1d_2_dilation,k2c_relu); 
k2c_pad1d(&conv1d_7_padded_input,&conv1d_6_output,conv1d_7_fill, 
	conv1d_7_pad); 
k2c_conv1d(&conv1d_7_output,&conv1d_7_padded_input,&conv1d_7_kernel, 
	&conv1d_7_bias,conv1d_7_stride,conv1d_7_dilation,k2c_relu); 
k2c_pad1d(&conv1d_3_padded_input,&conv1d_2_output,conv1d_3_fill, 
	conv1d_3_pad); 
k2c_conv1d(&conv1d_3_output,&conv1d_3_padded_input,&conv1d_3_kernel, 
	&conv1d_3_bias,conv1d_3_stride,conv1d_3_dilation,k2c_relu); 
k2c_pad1d(&conv1d_8_padded_input,&conv1d_7_output,conv1d_8_fill, 
	conv1d_8_pad); 
k2c_conv1d(&conv1d_8_output,&conv1d_8_padded_input,&conv1d_8_kernel, 
	&conv1d_8_bias,conv1d_8_stride,conv1d_8_dilation,k2c_relu); 
k2c_concatenate(&concatenate_5_output,concatenate_5_axis,concatenate_5_num_tensors0,&conv1d_1_output,&conv1d_2_output,&conv1d_3_output); 
k2c_concatenate(&concatenate_6_output,concatenate_6_axis,concatenate_6_num_tensors0,&conv1d_6_output,&conv1d_7_output,&conv1d_8_output); 
k2c_pad1d(&conv1d_4_padded_input,&concatenate_5_output,conv1d_4_fill, 
	conv1d_4_pad); 
k2c_conv1d(&conv1d_4_output,&conv1d_4_padded_input,&conv1d_4_kernel, 
	&conv1d_4_bias,conv1d_4_stride,conv1d_4_dilation,k2c_tanh); 
k2c_pad1d(&conv1d_9_padded_input,&concatenate_6_output,conv1d_9_fill, 
	conv1d_9_pad); 
k2c_conv1d(&conv1d_9_output,&conv1d_9_padded_input,&conv1d_9_kernel, 
	&conv1d_9_bias,conv1d_9_stride,conv1d_9_dilation,k2c_tanh); 
k2c_pad1d(&conv1d_5_padded_input,&conv1d_4_output,conv1d_5_fill, 
	conv1d_5_pad); 
k2c_conv1d(&conv1d_5_output,&conv1d_5_padded_input,&conv1d_5_kernel, 
	&conv1d_5_bias,conv1d_5_stride,conv1d_5_dilation,k2c_linear); 
k2c_pad1d(&conv1d_10_padded_input,&conv1d_9_output,conv1d_10_fill, 
	conv1d_10_pad); 
k2c_conv1d(&conv1d_10_output,&conv1d_10_padded_input,&conv1d_10_kernel, 
	&conv1d_10_bias,conv1d_10_stride,conv1d_10_dilation,k2c_linear); 
k2c_reshape(&conv1d_5_output,target_temp_newshp,target_temp_newndim); 
target_temp_output->ndim = conv1d_5_output.ndim; // copy data into output struct 
target_temp_output->numel = conv1d_5_output.numel; 
memcpy(target_temp_output->shape,conv1d_5_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
memcpy(target_temp_output->array,conv1d_5_output.array,target_temp_output->numel*sizeof(target_temp_output->array[0])); 
k2c_reshape(&conv1d_10_output,target_dens_newshp,target_dens_newndim); 
target_dens_output->ndim = conv1d_10_output.ndim; // copy data into output struct 
target_dens_output->numel = conv1d_10_output.numel; 
memcpy(target_dens_output->shape,conv1d_10_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
memcpy(target_dens_output->array,conv1d_10_output.array,target_dens_output->numel*sizeof(target_dens_output->array[0])); 

 } 

void etemp_profile_predictor_initialize() { 

} 

void etemp_profile_predictor_terminate() { 

} 

#endif /* ETEMP_PROFILE_PREDICTOR_H */
