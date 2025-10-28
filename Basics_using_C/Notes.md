What is a GPU:
GPU, or a Graphics processing unit is a specialized processor designed to handle highly parallel tasks, like "Embarrasingly parallel problems" that require a lot of simple arithmetic operations to be performed in a very short time. GPU work sby splitting a large problem into smaller problems that can be solved parallely.

Difference in CPU and GPU
CPU is desogned to solve complex problems serially, it can achieve only limited parallelism through its 4 or 8 cores. But, problems that require very simple arithmetic, like rendering a 3D object by processing the position of each vertex in the mesh of the object, does not require complex processing. 
If these problems are solved using the CPU, it takes a lot of time as the data to process is huge. Hence GPUs are used for this purpose achieving lesser processing time for data as compared to CPU for simple operations.

GPU architecture:
