## Simple Ray Tracer 

This project implements a ray tracer focusing on the core mechanics of ray generation and geometric intersection, excluding external light sources.

The primary goal is to understand the fundamentals of ray generation,  viewport projection, and ray–sphere intersection.
This is implemented using CPU. It will be optimised for GPU using CUDA in the next section.

---

### Overview

The camera is positioned to look along the positive Z-axis. Each object (sphere) is shaded purely based on its **distance from the viewport** (**depth shading**).

Overall Flow:

1.  Build a **viewport** from the camera and FOV.
2.  **Generate rays** for every pixel on the screen.
3.  Check **ray–sphere intersection** for all spheres.
4.  **Shade the pixel** based on the closest intersection distance (the *t* value).


---

### Data Structures used

#### `struct Point3`
Represents both a **3D point** and a **3D vector**.

* **Stores:** Three floats: $(x, y, z)$.
* **Purpose:** Used for positions, directions, and color values.
* **Vector Operations Supported:**
    * Addition and subtraction
    * Scalar multiplication and division
    * **Dot product**
    * **Magnitude** (vector length)


#### `struct Ray`
A ray in 3D space is defined by the parametric equation:
$$\mathbf{r}(t) = \text{origin} + t \cdot \text{direction}$$
* **`origin`**: The starting point of the ray.
* **`direction`**: A normalized direction vector (type `Point3`).

The parameter $t$ allows to get any point along the ray's path.

#### `struct Sphere`
Represents a sphere using:
* `center` Position in 3D space (type `Point3`).
* `radius` Sphere size (float).

---

### Ray–Sphere Intersection

The core mathematical task is to solve for the parameter $t$ when the ray equation is substituted into the sphere equation.
**The Intersection Function Calculates:**

1.  The **discriminant** ($\Delta$).
2.  The two possible $t$ values (potential intersection distances).
3.  **Returns:** The **smallest positive $t$** (the nearest visible intersection).
4.  **Returns:** A large constant if the discriminant is negative (no real hit occurs).

---

### Program Flow

#### 1. Set up the fundamental configuration values
This includes the final image pixel dimensions, the viewport size calculated from the field of view and aspect ratio, and the camera's location in the scene.

#### 2. Create PPM Output File
The renderer outputs a simple **PPM (P3)** image format, which stores ASCII RGB values. This choice avoids external libraries, keeping the program minimal and self-contained.

#### 3. Generate Random Spheres
A set of spheres is created randomly:
* Placed in front of the camera.
* Positioned within the bounds of the viewport width/height.
* Given small, random radii.

#### 4. Ray Generation per Pixel
For each pixel at coordinates $(i, j)$:
1.  Convert $(i, j)$ into normalized viewport coordinates $(u, v)$.
2.  Map $(u, v)$ to a real 3D viewport point $(x_{\text{pix}}, y_{\text{pix}})$.
3.  Construct a **Ray** from the camera origin through $(x_{\text{pix}}, y_{\text{pix}})$.
4.  **Intersect** the ray with every sphere in the scene.
5.  Select the **closest (smallest positive) $t$ value**.

#### Depth-Based Shading

Since no light sources are used, the shading is based solely on the distance ($t$):

* **Closer spheres** (smaller $t$) appear **brighter** (closer to 255).
* **Farther spheres** (larger $t$) appear **darker** (closer to 0).

This produces a clean grayscale image with a clear 3D depth illusion.

---

### Final Output
<img width="512" height="512" alt="output_img_1" src="https://github.com/user-attachments/assets/20d9e7e0-0468-47d9-9ab6-3ef829a68828" />


