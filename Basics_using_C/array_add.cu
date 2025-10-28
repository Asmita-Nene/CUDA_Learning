#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void addArrays(int * arr1, int* arr2, int* res){
    int tid = threadIdx.x;  
    if (tid < 5) {          // boundary check
        res[tid] = arr1[tid] + arr2[tid];
    }
    printf("Inside the GPU\n");
}

int main(){
    int arr1[5] = {1, 2, 3, 4, 5};
    int arr2[5] = {10, 20, 30, 40, 50};
    int res_arr[5];

    int* arr1_device_ptr;
    int* arr2_device_ptr;
    int* res_arr_device_ptr;

    // Allocate GPU memory
    cudaMalloc(&arr1_device_ptr, 5*sizeof(int));
    cudaMalloc(&arr2_device_ptr, 5*sizeof(int));
    cudaMalloc(&res_arr_device_ptr, 5*sizeof(int));

    // Copy arrays to GPU
    cudaMemcpy(arr1_device_ptr, arr1, 5*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(arr2_device_ptr, arr2, 5*sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    addArrays<<<1, 5>>>(arr1_device_ptr, arr2_device_ptr, res_arr_device_ptr);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(res_arr, res_arr_device_ptr, 5*sizeof(int), cudaMemcpyDeviceToHost);

    // Print result properly
    cout << "The result is:\n";
    for(int i = 0; i < 5; i++)
        cout << res_arr[i] << " ";
    cout << endl;

    // Free GPU memory
    cudaFree(arr1_device_ptr);
    cudaFree(arr2_device_ptr);
    cudaFree(res_arr_device_ptr);

    return 0;
}
