#include <stdio.h>
#include <math.h>
#include <stdarg.h>
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


void model(k2c_tensor A, k2c_tensor B, int reset_states) {

  static float lstm_state[20];
  if (reset_states) {
    memset(lstm_state,0,20*sizeof(float));
  }



}









































void k2c_matmul(float C[], float A[], float B[], size_t outrows,
		size_t outcols, size_t innerdim) ;
void printer(float x[], int size);
void k2c_affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
		       size_t outcols, size_t innerdim);

void relu(float x[], size_t size);
void linear(float x[], size_t size);
void dense(float output[], float input[], float kernel[], float bias[],
	   size_t outrows, size_t outcols, size_t innerdim,
	   void (*activation) (float[], size_t));
void min(float output[], size_t size, size_t num,...);
float keras2c_arrmax(float array[], size_t numels, size_t offset);
void tensor_copy(k2c_tensor* bar);
void k2c_exponential(float x[], size_t size);
void wrapper(k2c_tensor* C, k2c_tensor* A, k2c_tensor* B, float work[]);
void wrapper2(k2c_tensor* C, k2c_tensor* A, k2c_tensor* B, float work[]);
void malloc_test(float* array);
  
int main(){
  float x1[12] = { 1,   2,3,  4,  5,    6,  7,   8  , 9,10,11,12};
  float x2[12] = {-3.2, 5,1,345,100.2,-13.1, .25,.01,-4,32, 8, 7.5};
  float x3_array[12] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
  float y_array[8] = {1,-2,3,-4,5,-6,7,-8};
  float d[2] = {1.23,4.56};
  float C_array[6];
  float output[12] = {0};
  int outrows = 3;
  int outcols = 2;
  float innerdim = 4;
  float a = -HUGE_VAL;
  float e[4] = {1,2,3,4,};
  float x[] = {1,-3,5.5,-HUGE_VAL};
  int size = 12;
  int rows = 3;
  int cols = 4;
  float work[20] = {0};
  k2c_tensor C = {&C_array[0],2,6,{2,3}};
  k2c_tensor x3 = {&x3_array[0],2,12,{3,4}};
  k2c_tensor y = {&y_array[0],2,8,{4,2}};

  
  /* for (int i=0; i<12; i++) { */
  /*   printf("%e \n", x3[i]);} */
  malloc_test(x1);

  
 /*  size_t in_height = 10; */
/*   for (int i=in_height-1; i>-1; i--) { */
/*     printf("%d %d \n",i,in_height-1-i);} */
  
/*   float foo_array[20] = { */
/* 7.6266743486e-01,8.5218802371e-01,7.4291503115e-01,7.5633826592e-01, */
/* 3.9278868836e-01,9.5637243036e-01,2.7966619768e-01,3.6879664082e-01, */
/* 4.1681139614e-01,1.6773200337e-01,8.8033095363e-01,2.7210711410e-01, */
/* 5.9513055263e-01,6.3869915141e-01,9.7067075876e-01,7.4522502229e-01, */
/* 2.5196224070e-01,9.5380225115e-01,3.2237026037e-02,7.1052879207e-02, */
/* }; */
  
/*   k2c_tensor foo = {&foo_array[0], 2,20,{4,5,1,1}}; */
/*   k2c_tensor bar; */
/*   k2c_tensor gah; */
/*   memcpy(&gah,&foo,sizeof(k2c_tensor)); */
/*     printf("%lu, %lu, %lu, %f, %f, %f \n",gah.ndim, gah.numel, gah.shape[1], gah.array[0], */
/* 	 gah.array[4], gah.array[19]); */
/*   tensor_copy(&bar); */
/*   printf("%lu \n", sizeof(foo)); */
/*   printf("%lu \n", sizeof(k2c_tensor)); */
/*   printf("%lu \n", sizeof(size_t)); */
/*   printf("%lu \n", sizeof(float*)); */
/*     printf("%lu, %lu, %lu, %f, %f, %f \n",bar.ndim, bar.numel, bar.shape[1], bar.array[0], */
/* 	 bar.array[4], bar.array[19]); */


//min(output,12,3,x1,x2,x3);

 /* for (size_t i=0; i<foo.shape[0]; i++){ */
 /*   printf("%f ", foo.array[i]);} */

  //dense(C,x,y,d,outrows,outcols,innerdim,linear);
  
return 0;
}

void malloc_test(float* array) {
  k2c_tensor foo = {array,1,12,{12,1,1,1}};
  for (size_t i=0; i<foo.numel; i++) {
    printf("%f \n", foo.array[i]);
  }
}

void wrapper(k2c_tensor* C, k2c_tensor* A, k2c_tensor* B, float work[])  {
  wrapper2(C,A,B,work);
}
  
void wrapper2(k2c_tensor* C, k2c_tensor* A, k2c_tensor* B, float work[]) {
  size_t outrows = 3;
  size_t outcols = 2;
  size_t innerdim = 4;
  float *work1 = &work[0];
  float *work2 = &work[12];

  for (int i=0; i<12; i++){
    work1[i] = A->array[i];}
  for (int i=0; i<8; i++){
    work2[i] = B->array[i];}  
  k2c_matmul(C->array,work1,work2,outrows,outcols,innerdim);
}

void passby(int a) {
  a = 0;
}

void k2c_exponential(float x[], size_t size){
  /* exponential activation */
  /* y = exp(x) */
  /* x is overwritten with the activated values */

  for (size_t i=0; i<size; i++) {
    x[i] = exp(x[i]);}
}

void tensor_copy(k2c_tensor* bar) {
    float foo_array[20] = {
7.6266743486e-01,8.5218802371e-01,7.4291503115e-01,7.5633826592e-01,
3.9278868836e-01,9.5637243036e-01,2.7966619768e-01,3.6879664082e-01,
4.1681139614e-01,1.6773200337e-01,8.8033095363e-01,2.7210711410e-01,
5.9513055263e-01,6.3869915141e-01,9.7067075876e-01,7.4522502229e-01,
2.5196224070e-01,9.5380225115e-01,3.2237026037e-02,7.1052879207e-02,
}; 
  k2c_tensor foo = {&foo_array[0], 2,20,{4,5,1,1}};
  memcpy(bar,&foo,sizeof(k2c_tensor));
}

void k2c_matmul(float C[], float A[], float B[], size_t outrows,
	    size_t outcols, size_t innerdim) {
  /* Just your basic 1d matrix multiplication. Takes in 1d arrays
 A and B, results get stored in C */
  /*   Size A: outrows*innerdim */
  /*   Size B: innerdim*outcols */
  /*   Size C: outrows*outcols */

  // make sure output is empty
        printf("in matmul \n");

  memset(C, 0, outrows*outcols*sizeof(C[0]));
  printf("did memset \n");

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

void k2c_affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
	    size_t outcols, size_t innerdim) {
  /* Computes C = A*B + d, where d is a vector that is added to each
 row of A*B*/
  /*   Size A: outrows*innerdim */
  /*   Size B: innerdim*outcols */
  /*   Size C: outrows*outcols */
  /*   Size d: outrows */

  // make sure output is empty

  printf("in affinematmul \n");
  memset(C, 0, outrows*outcols*sizeof(C[0]));
  printf("did memset \n");

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

void printer(float x[], int size)  {

  for (int i=0; i<size; i++) {
    printf("%f \n", x[i]);
    }
  
  printf("\n");
}

void relu(float x[], size_t size) {
  /* Rectified Linear Unit activation (ReLU) */
  /*   y = max(x,0) */
  /* x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = 0.0f;
    }
  }
}

 void linear(float x[], size_t size){
  /* linear activation. Doesn't do anything, just a blank fn */
  printf("in linear");
}



void min(float output[], size_t numels, size_t numtensors,...){

  va_list args;
  float *arrptr;
  va_start (args, numtensors);
  arrptr = va_arg(args, float*);

  for (size_t i=0;i<numels;i++){
    output[i] = arrptr[i];}
  
  for (size_t i = 0; i < numtensors-1; i++){
    arrptr = va_arg(args, float*);
    for (size_t j=0; j<numels; j++){
      if (output[j]>arrptr[j]){
	output[j] = arrptr[j];}
    }
  }
  va_end (args);             
}

float keras2c_arrmax(float array[], size_t numels, size_t offset){

  float max=array[0];
  for (size_t i=1; i<numels; i++){
    if (array[i*offset]>max){
      max = array[i*offset];}
  }
  return max;
}
