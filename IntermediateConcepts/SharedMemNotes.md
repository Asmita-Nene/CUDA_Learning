# Shared Memory in CUDA

**Shared memory** is a fast, on-chip scratchpad memory located within each **Streaming Multiprocessor (SM)**. It allows threads within the **same block** to access and share data efficiently.

## Key Characteristics

- **Scope:** Only threads within the same block can access the shared memory. Threads from other blocks **cannot** modify or access it.
- **Fast Access:** Shared memory resides on the SM itself, not in VRAM, making it much faster than global memory.
- **Concurrency:** An SM can run multiple blocks concurrently by executing warps from different blocks simultaneously. Despite this, each block’s shared memory remains isolated and accessible only to its own threads.
- **Memory Limitation:** Shared memory is limited in size. If a kernel uses more shared memory per block, fewer blocks can run concurrently on the same SM.  
  **Implication:** Higher shared memory usage , fewer concurrent blocks and hence may affect overall parallelism and performance.

#Implementation using C/C++
## CUDA Dot Product Using Shared Memory

This program computes the **dot product of two vectors** using CUDA.  
It demonstrates how to use **shared memory** for fast intra-block communication and parallel reduction to sum partial results efficiently. The executable program is attached in this same folder.
### Important Points from the Program

---

 `__shared__ float shared[THREADS_PER_BLOCK];`
Declares an array in **shared memory**, sized according to the number of threads per block.  
Each block gets **its own private copy** of this array in the GPU’s shared memory.

Used to store intermediate (partial) results — for example, products or partial sums — that can be accessed and updated by all threads within the same block.

---

`__syncthreads();`
This function **synchronizes all threads in a block**.  
Execution of every thread pauses at this point until all threads in the same block have reached it.

During reduction or shared memory operations:
- Each thread updates values in shared memory.
- Before moving to the next step, all writes must be completed and visible to other threads.
- `__syncthreads()` ensures that **no thread proceeds** before all others have finished their writes.

**Prevents:**  
- **Data races** (when multiple threads read/write shared memory simultaneously without coordination).
- **Incorrect results** due to partial updates.

**Important Rule:**  
`__syncthreads()` **must be reached by all threads** in a block.  
It **must not** be placed inside a conditional statement that some threads might skip.  

--- 
`atomicAdd()` Function: 
This is a CUDA intrinsic function used to **safely perform addition operations on shared or global memory** when multiple threads may attempt to update the same variable simultaneously.
Here, it adds the local blockwise additions to the global result, one by one. 
