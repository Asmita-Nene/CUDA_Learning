# Simple Raytracer Using GPU 

This raytracer uses CUDA to accelerate image generation by executing the same raytracing logic as the CPU version, but in parallel across thousands of GPU threads. The rendering algorithm is the same as the CPU version; only the execution model is parallelized.  
The following sections summarize the core CUDA concepts used in this implementation.

---

## 1. Bitmaps

The rendered image is stored using a bitmap array, which is later written into a PPM file.  
Since GPU execution is asynchronous, threads cannot safely write to a file directly. Instead, each thread writes its pixel color into a bitmap stored in GPU memory.

### Why Bitmaps?

- A bitmap lets the GPU store the image in memory before transferring it back to the CPU.
- Even though images are 2D, they are stored as a 1D array in memory for efficient access.

### Structure Used

`uchar3` stores `{R, G, B}` as three unsigned bytes — perfect for pixel data.

### Indexing Formula

To map a 2D `(x, y)` pixel to the 1D array: idx = y * width + x


Each CUDA thread computes `(x, y)` and writes its pixel value to this index.

---

## 2. 2D Thread Blocks

CUDA organizes threads into blocks, and blocks into a grid.  
Most GPUs allow up to **1024 threads per block**.

To process a 2D image, we use **2D blocks** so that threads map naturally to pixel coordinates.

- Direct thread-to-pixel mapping  
- Cleaner and more intuitive indexing  
- Avoids manually converting 1D thread indices into 2D coordinates  
- Often improves memory access patterns for image workloads  

---

## 3. Constant Memory

GPU constant memory is a small, cached, read-only region ideal for storing data that:

- does not change during kernel execution  
- is accessed by many threads  

All threads need access to the sphere data when performing ray–sphere intersection tests.  
Placing the sphere array in `__constant__` memory allows the GPU to broadcast these reads efficiently.

This improves performance compared to reading from global memory repeatedly.

---

## 4. `__device__`, `__host__`

CUDA uses function qualifiers to specify where a function runs and where it can be called from.

### `__host__`

- Function runs on the **CPU**
- Can be called only from **host (CPU) code**
- This is the default; if no qualifier is given, the function is `__host__` by default

### `__device__`

- Function runs on the **GPU**
- Can be called only from **device (GPU) code**, except from kernels
- Device functions are not entry points like kernels; they must be called from another GPU function.

---
## Output Image
<img width="512" height="512" alt="raytracing_op (5)" src="https://github.com/user-attachments/assets/9fc4db98-9807-4bb2-a2b5-8f1cf7634f89" />

