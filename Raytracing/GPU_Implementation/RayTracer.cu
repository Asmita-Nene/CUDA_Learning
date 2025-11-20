#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#define M_PI 3.14159265
using namespace std;
 const int num_spheres = 20;


// 3D Vector 
struct Point3 {
    float x, y, z;
    __host__ __device__ Point3() {}
    __host__ __device__ Point3(float xin, float yin, float zin) : x(xin), y(yin), z(zin) {}

    __host__ __device__ Point3 operator + (const Point3& v) const { return Point3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Point3 operator - (const Point3& v) const { return Point3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Point3 operator * (float t) const { return Point3(x*t, y*t, z*t); }
   __host__ __device__  Point3 operator / (float t) const { return Point3(x/t, y/t, z/t); }

    __host__ __device__ float dot(const Point3& v) const { return x*v.x + y*v.y + z*v.z; }
    __host__ __device__ float magnitude() const { return sqrtf(x*x + y*y + z*z); }
};

// Ray
struct Ray {
    Point3 origin;
    Point3 direction;

    __host__ __device__ Ray() {}
   __host__ __device__  Ray(Point3 o, Point3 d) : origin(o), direction(d / d.magnitude()) {}
};

// Sphere
struct Sphere {
    Point3 center;
    float radius;
};

 __constant__ Sphere dev_spheres[num_spheres];
// Random float 0 → 1
float randf() { 
    return rand() / (float)RAND_MAX; 
}

// Sphere-ray intersection: returns t along ray, or large number if no hit
__device__ float intersectSphere(const Ray& ray, const Sphere& sphere) {
    Point3 L = ray.origin - sphere.center;
    float a = ray.direction.dot(ray.direction);
    float b = 2 * L.dot(ray.direction);
    float c = L.dot(L) - sphere.radius * sphere.radius;
    float det = b*b - 4*a*c;


    if(det < 0) return 1e9; // no intersection

    float sqrt_det = sqrtf(det);
    float t0 = (-b - sqrt_det) / (2*a);
    float t1 = (-b + sqrt_det) / (2*a);

    if(t0 > 0) return t0;
    if(t1 > 0) return t1;
    return 1e9;
}


__global__ void paintImage(uchar3 *bitmap, int width, int height, float vp_width, float vp_height, Point3 camera){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width || y >= height){
        return;
    }

    float u = float(x)/(width-1);
    float v = float(y)/(height-1);
    float x_pix = (u-0.5f)*vp_width;
    float y_pix = (0.5f-v)*vp_height;

    Point3 ray_dir(x_pix - camera.x, y_pix - camera.y, 0 - camera.z);
    Ray ray(camera, ray_dir);


    float closest_t = 1e9;
    for(int k=0; k<20; k++){
        float t = intersectSphere(ray, dev_spheres[k]);
        if(t < closest_t) closest_t = t;
    }
    float val = 0;
    if(closest_t < 1e9){
        float max_depth = 10.0f;
        val = 255 * (1.0f - closest_t / max_depth);
        val = max(0.0f, min(255.0f, val));
    }

    int iv = int(val);
    int idx = y * width + x;
    bitmap[idx].x = iv;
    bitmap[idx].y = iv;
    bitmap[idx].z = iv;
   
}

void writePPM(uchar3 *bitmap, int width, int height, string filename){
  ofstream img(filename);
  img << "P3\n" << width << " " << height << "\n" << 255 << "\n";
  int idx;

  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
        idx = i * width + j;
        img << (int)bitmap[idx].x << " "
            << (int)bitmap[idx].y << " "
            << (int)bitmap[idx].z << " ";
    }
    img << "\n";
}

  img.close();
}


int main() {
    // Image parameters
    const int img_width = 512;
    const int img_height = 512;
    string filename = "raytracing_op.ppm";

    // Viewport /camera 
    float f = 1.0f; // distance to viewport
    float FOV_y = 90.0f;
    float aspect = float(img_width)/img_height;
    float vp_height = 2*f*tan((FOV_y/2) * M_PI/180.0f);
    float vp_width = vp_height * aspect;

    Point3 camera(0, 0, -1);

    //GPU setup
    int bitmap_size = img_width * img_height * sizeof(uchar3);
    uchar3 *host_bitmap = (uchar3*) malloc(bitmap_size);
    uchar3 *bitmap;
    cudaMalloc(&bitmap, bitmap_size);

    dim3 block(32, 32);
    dim3 grid(((img_width + block.x - 1)/block.x), ((img_height + block.y - 1)/block.y));
 
    // Random sphres generation
    Sphere spheres[num_spheres];
    for(int i=0;i<num_spheres;i++){
        spheres[i].center.x = ((randf())*(vp_width)) - (vp_width/2);
        spheres[i].center.y = ((randf())*vp_height) - (vp_height/2);
        spheres[i].center.z = 2.0f + randf()*8.0f; // in front of camera
        spheres[i].radius = 0.1f + randf()*0.5f; // small radius
    }
   
    cudaMemcpyToSymbol(dev_spheres, spheres, sizeof(Sphere) * num_spheres);

    paintImage<<<grid, block>>>(bitmap, img_width, img_height, vp_width, vp_height, camera);
    cudaDeviceSynchronize();

    cudaMemcpy(host_bitmap, bitmap, bitmap_size, cudaMemcpyDeviceToHost);

    writePPM(host_bitmap, img_width, img_height, filename);

    return 0;
}
