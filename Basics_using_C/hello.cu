#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void hello_msg(){
  printf("Hello from CUDA and GPU\n");
}

int main(){
  cout<<"Hello this is the start of the host code, now calling the kernel\n";
  hello_msg<<<1, 32>>>();

  cudaDeviceSynchronize();
  return 0;
}