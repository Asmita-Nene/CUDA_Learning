# What is a GPU

A **GPU (Graphics Processing Unit)** is a specialized processor designed to handle **highly parallel tasks**, such as *"embarrassingly parallel problems"* that require performing a large number of simple arithmetic operations in a very short time.

GPUs work by **splitting a large problem into smaller problems** that can be solved **in parallel**, allowing for much faster processing of tasks that can be executed simultaneously.


# Difference Between CPU and GPU

- **CPU (Central Processing Unit)** is designed to solve **complex problems serially**. It typically has **4 to 8 cores**, providing only limited parallelism.

- **GPU** is optimized for tasks that involve **simple arithmetic operations across large datasets**, such as rendering a 3D object by processing the position of each vertex in a mesh.

- Using a CPU for such tasks takes **much longer** because of the sheer volume of data and the simple operations involved. GPUs, by executing these operations in parallel, can **process large amounts of data much faster** than a CPU for these specific types of workloads.

# GPU architecture:


## Diagram of a Typical GPU Architecture
<img width="720" height="683" alt="Screenshot 2025-10-28 231056" src="https://github.com/user-attachments/assets/e2825b93-be90-4a7a-b5ee-733002c10eb2" />

*Source: S. Narayanan, M. Govindaraju, and S. K. Nair, "Characterizing the Microarchitectural Implications of a Convolutional Neural Network (CNN) Execution on GPUs," ResearchGate, 2018, Figure 3. Used for personal learning purposes.*

# GPU Architecture

A modern GPU is composed of several key components that work together to achieve massive parallelism and high-performance computation. Below is an overview of the main elements:

---

## Streaming Multiprocessor (SM)

- **Definition:** The fundamental computing unit of a GPU.  
- **Components / Contains:** Multiple CUDA cores, tensor cores, and shared memory.  
- **Use:** Executes parallel workloads by managing threads and coordinating computation across its cores.

---

## CUDA Cores (SP Units)

- **Definition:** Basic arithmetic units inside an SM, also called **Streaming Processors (SPs)**.  
- **Use:** Perform simple arithmetic operations such as addition, multiplication, and logical operations.  
- **Purpose:** Ideal for highly parallel, compute-intensive tasks like vector operations and graphics rendering.

---

## Tensor Cores

- **Definition:** Specialized cores designed specifically for **matrix operations**.  
- **Use:** Accelerate deep learning computations, such as matrix multiplications used in neural network training and inference.  

---

## Memory Units

- **VRAM (Global Memory):** Main memory accessible to all SMs. Stores large datasets and kernel inputs/outputs.  
- **L2 Cache:** High-speed cache shared among SMs, helps reduce memory latency.  
- **Shared Memory:** Fast, small memory local to each SM. Allows threads within the same block to communicate and share data efficiently.  

---

# Introduction to CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform by NVIDIA that lets you use GPUs for general-purpose computing.



## Prerequisites

Before learning CUDA, it helps to understand the **hierarchy of execution units** in a GPU:



## Threads

- **Smallest unit of execution** in CUDA.  
- Runs on **one CUDA core** of a Streaming Multiprocessor (SM).  
- Each thread works **independently** and has its own **registers and private data**.

---

## Blocks

- A **group of threads** that can **share data using shared memory**.  
- Each block can have up to **1024 threads** (may vary depending on GPU).  
- Threads inside a block can **cooperate** efficiently.

---

## Grid

- A **collection of blocks** representing the **whole problem** the GPU is solving.  
- Blocks in a grid are **independent** and can run on different SMs.

---

## Warp

- A group of **32 threads** that execute the **same instruction at the same time** (SIMD).  
- The GPU schedules and executes **warps**, not individual threads.  
- For best performance, the **total number of threads should be a multiple of 32**.

---

## Notes on Execution

- **One CUDA core** executes **one thread completely**.  
- **One block** runs entirely on **one SM**; its warps do not move to other SMs.  
- **One SM** can handle **multiple blocks** by interleaving warps from these blocks.

# CUDA Programming

---

## Kernels

- A **kernel** is a function that runs on the **GPU** instead of the CPU.  
- It can be launched with **many threads in parallel**.  
- The GPU executes **one kernel at a time**, but across **multiple threads**.

---

## Typical CUDA Programming Workflow

1. **Load data** into the CPU memory.  
2. **Copy data** from CPU memory to GPU memory.  
3. **Launch the kernel** on the GPU.  
4. **Copy the results** from GPU memory back to CPU memory.  
5. **Use or display the result** on the CPU.

---

## Why We Need to Copy Data Between CPU and GPU

- The **CPU and GPU have separate memory spaces**.  
- The GPU **cannot directly access** the CPU’s memory, and vice versa.  
- Therefore, data must be **explicitly copied**:
  - From CPU → GPU (before computation)  
  - From GPU → CPU (after computation)
## cudaMemcpy()

The `cudaMemcpy()` function is used to **copy data between CPU (host) memory and GPU (device) memory**.

---

### **Syntax**

```cpp
 cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
```
The `cudaMemcpyKind` parameter specifies **the direction of data transfer** between CPU and GPU memory spaces.

## Example: A Simple CUDA Program

This example shows how to write a basic CUDA program that runs a kernel on the GPU.

```cpp
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void hello_msg() {
    printf("Hello from CUDA and GPU\n");
}

int main() {
    cout << "Hello this is the start of the host code, now calling the kernel\n";
    hello_msg<<<1, 32>>>();

    cudaDeviceSynchronize();
    return 0;
}
```
## Explanation

#### 1. __global__ keyword

- The __global__ keyword tells the compiler that the function runs on the GPU (device) but is called from the CPU (host).
- In this case, hello_msg() is a CUDA kernel, meaning it can be executed by many threads in parallel on the GPU.

#### 2. Launching the CUDA Kernel 
hello_msg<<<1, 32>>>();


- The triple angle brackets <<< >>> are used to launch the kernel.
- The parameters inside the brackets specify how many blocks and threads per block to launch:
-- 1 → number of blocks
-- 32 → number of threads per block
So, this launches 32 threads (1 block × 32 threads) that all run the hello_msg() function in parallel on the GPU.

#### 3. cudaDeviceSynchronize()

By default, kernel launches are asynchronous, meaning the CPU does not wait for the GPU to finish.

cudaDeviceSynchronize() is used to make the CPU wait until the GPU has completed all work.
- Without it, the program might end before the GPU finishes printing or processing.


