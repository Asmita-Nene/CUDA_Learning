## Ray Tracing

Ray tracing is a powerful **rendering technique** that aims to realistically simulate the lighting of a scene by modeling the physical behavior of light. This results in highly realistic images featuring physically accurate reflections, shadows, refractions, and light bouncing (indirect illumination).

### Working of Ray Tracing

The core steps for determining the color of a single pixel are:

#### 1. Ray Casting 
A **primary ray** (or view ray) is cast from the virtual camera through the center of each pixel on the screen into the 3D scene.

#### 2. Intersection 
The algorithm determines which object the primary ray intersects first.
Acceleration: To efficiently handle complex geometry, algorithms like Bounding Volume Hierarchies (BVH) are used to speed up the intersection test.


#### 3. Secondary Rays 
From the point of intersection on the object's surface, one or more secondary rays are cast to gather light information:

| Ray Type | Target | Purpose |
| :--- | :--- | :--- |
| **Shadow Rays** | Light Sources | Determine if the point is in shadow (if blocked). |
| **Reflection Rays** | Scene | Calculate **reflected light** from surrounding objects (for shiny surfaces). |
| **Refraction Rays** | Through Surface | Calculate how light bends when passing through transparent materials. |

#### 4. Color Determination 
The final pixel color is calculated based on:
* The object's properties (material, color, texture).
* The illumination information collected by all rays.

#### Problem: Complex Intersections
Finding the exact intersection point of a ray with complex, irregular surfaces (like a human face) using its precise mathematical formula is computationally expensive.

#### Solution: Triangulation
Modern renderers divide 3D objects into small triangles (a mesh).
* BVH identifies the small group of triangles intersected by the ray.
* The intersection and light calculation are performed only on that specific group of triangles.

#### Implementation Note
For simplicity, this implementation directly uses the mathematical formula of a sphere for intersection. The initial version is CPU-based, with subsequent optimizations developed for the GPU using CUDA.
