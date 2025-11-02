#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 32    //chosen as multiple of 32, as SM executes one warp at a time, i.e group of 32 threads
#define N 1024

__global__ void dotProduct(float* a, float* b, float * result){

  __shared__ float cache[THREADS_PER_BLOCK];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int local_tid = threadIdx.x;
  float temp = 0.00f;

  //Getting product of individual elements of the array, and storing them into the cache array
  if(tid < N){
    temp = a[tid] * b[tid];
  }
  cache[local_tid] = temp;
  __syncthreads();

  //reduction - adding the products in the block - adding elements of the cache array in the shared memory
  int stride = blockDim.x/2;
  while(stride != 0){
    if(local_tid < stride){
      cache[local_tid] = cache[local_tid] + cache[local_tid + stride];
    }
    __syncthreads();
    stride = stride / 2;
  }

  //adding the local summations from different blocks
  if(local_tid == 0){//to ensure only one thread accesses the sum from one block
    atomicAdd(result, (float)cache[0]);
  }

}

int main(){
  //declare host variables
  float arr1[N];
  float arr2[N];
  float result = 0.0;
  float zero = 0.0;

  //define the arrays(cons - runs on CPU )- improvement - define another kernel to generate random numbers and populate the array
  srand(time(NULL));
  float min = 0.00f;
  float max = 10.00f;
  for(int i = 0; i < N; i ++){
    arr1[i] = min + ((float)rand() / RAND_MAX) * (max - min);
    arr2[i] = min + ((float)rand() / RAND_MAX) * (max - min);
  }

  //Copy data to GPU
  float* arr1_device_ptr;
  float* arr2_device_ptr;
  float* res_device_ptr;

  cudaMalloc(&arr1_device_ptr, N*sizeof(float));
  cudaMalloc(&arr2_device_ptr, N*sizeof(float));
  cudaMalloc(&res_device_ptr, sizeof(float));

  cudaMemcpy(arr1_device_ptr, arr1, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(arr2_device_ptr, arr2, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(res_device_ptr, &zero, sizeof(float), cudaMemcpyHostToDevice);


  //call the kernel
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dotProduct<<<blocks, THREADS_PER_BLOCK>>>(arr1_device_ptr, arr2_device_ptr, res_device_ptr);
  cudaDeviceSynchronize();

  //copy result to the CPU
  cudaMemcpy(&result, res_device_ptr, sizeof(float) ,cudaMemcpyDeviceToHost);

  //display
  std::cout<<"The result is = "<<result;

  //free the memory
  cudaFree(arr1_device_ptr);
  cudaFree(arr2_device_ptr);
  cudaFree(res_device_ptr);


  return 0;
}