#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void addNums(int a, int b, int* res){
  *res = a + b;
  printf("Added the numbers in this function inside the GPU\n");
}

int main(){
  int a = 10;
  int b = 15;
  int res = 0;

  int*  device_res_ptr;

  cudaMalloc(&device_res_ptr, sizeof(int));

  addNums<<<1, 1>>>(a, b, device_res_ptr);
  cudaDeviceSynchronize();

  cudaMemcpy(&res, device_res_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  cout<<"\nResult is: "<<res<<endl;

  cudaFree(device_res_ptr);

  return 0;
}