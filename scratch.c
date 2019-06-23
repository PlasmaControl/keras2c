#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>


struct k2c_tensor
{
  float *array;
  size_t shape[4];
  size_t ndims;
};

void printer(float x[], int size);
void affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
		   size_t outcols, size_t innerdim);
void relu(float x[], size_t size);
void linear(float x[], size_t size);
void dense(float output[], float input[], float kernel[], float bias[],
	   size_t outrows, size_t outcols, size_t innerdim,
	   void (*activation) (float[], size_t));
void min(float output[], size_t size, size_t num,...);
float keras2c_arrmax(float array[], size_t numels, size_t offset);

int main(){
  float x1[12] = { 1,   2,3,  4,  5,    6,  7,   8  , 9,10,11,12};
  float x2[12] = {-3.2, 5,1,345,100.2,-13.1, .25,.01,-4,32, 8, 7.5};
  float x3[12] = { 5,   5,7,  4,  5,    7,  4,  67,   5, 4, 7,63};
  float y[8] = {1,-2,3,-4,5,-6,7,-8};
  float d[2] = {1.23,4.56};
  float C[6];
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


  float foo_array[20] = {
7.6266743486e-01,8.5218802371e-01,7.4291503115e-01,7.5633826592e-01,
3.9278868836e-01,9.5637243036e-01,2.7966619768e-01,3.6879664082e-01,
4.1681139614e-01,1.6773200337e-01,8.8033095363e-01,2.7210711410e-01,
5.9513055263e-01,6.3869915141e-01,9.7067075876e-01,7.4522502229e-01,
2.5196224070e-01,9.5380225115e-01,3.2237026037e-02,7.1052879207e-02,
}; 
struct k2c_tensor foo = {&foo_array[0], {4,5,1,1},2};

 
//min(output,12,3,x1,x2,x3);

 for (size_t i=0; i<foo.shape[0]; i++){
   printf("%f ", foo.array[i]);}

  //dense(C,x,y,d,outrows,outcols,innerdim,linear);
  //  affine_matmul(C,x,y,d,outrows,outcols,innerdim);
  /* if (-400000>a){ */
  /*   printer(output,12);} */
  /* for (int i=0; i< 5; i++){ */
  /*   for (int j=0;j<4;j++) { */
  /*     printf("%f ",z[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  
return 0;
}


void affine_matmul(float C[], float A[], float B[], float d[], size_t outrows,
	    size_t outcols, size_t innerdim) {
  /* Computes C = A*B + d, where d is a vector that is added to each
 row of A*B*/
  /*   Size A: outrows*innerdim */
  /*   Size B: innerdim*outcols */
  /*   Size C: outrows*outcols */
  /*   Size d: outrows */

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

void dense(float output[], float input[], float kernel[], float bias[],
	   size_t outrows, size_t outcols, size_t innerdim,
	   void (*activation) (float[], size_t)){
  printf("in dense");
  size_t outsize = outrows*outcols;
  affine_matmul(output,input,kernel,bias,outrows,outcols,innerdim);
  activation(output,outsize);
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
