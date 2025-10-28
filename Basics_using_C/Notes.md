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

