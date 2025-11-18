#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#define M_PI 3.14159265
using namespace std;

// 3D Vector 
struct Point3 {
    float x, y, z;
    Point3() {}
    Point3(float xin, float yin, float zin) : x(xin), y(yin), z(zin) {}

    Point3 operator + (const Point3& v) const { return Point3(x + v.x, y + v.y, z + v.z); }
    Point3 operator - (const Point3& v) const { return Point3(x - v.x, y - v.y, z - v.z); }
    Point3 operator * (float t) const { return Point3(x*t, y*t, z*t); }
    Point3 operator / (float t) const { return Point3(x/t, y/t, z/t); }

    float dot(const Point3& v) const { return x*v.x + y*v.y + z*v.z; }
    float magnitude() const { return sqrt(x*x + y*y + z*z); }
};

// Ray
struct Ray {
    Point3 origin;
    Point3 direction;

    Ray() {}
    Ray(Point3 o, Point3 d) : origin(o), direction(d / d.magnitude()) {}
};

// Sphere
struct Sphere {
    Point3 center;
    float radius;
};

// Random float 0 → 1
float randf() { return rand() / (float)RAND_MAX; }

// Sphere-ray intersection: returns t along ray, or large number if no hit
float intersectSphere(const Ray& ray, const Sphere& sphere) {
    Point3 L = ray.origin - sphere.center;
    float a = ray.direction.dot(ray.direction);
    float b = 2 * L.dot(ray.direction);
    float c = L.dot(L) - sphere.radius * sphere.radius;
    float det = b*b - 4*a*c;

    if(det < 0) return 1e9; // no intersection

    float sqrt_det = sqrt(det);
    float t0 = (-b - sqrt_det) / (2*a);
    float t1 = (-b + sqrt_det) / (2*a);

    if(t0 > 0) return t0;
    if(t1 > 0) return t1;
    return 1e9;
}

int main() {
    // Image parameters
    const int img_width = 512;
    const int img_height = 512;
    const int max_col_val = 255;

    // Viewport /camera 
    float f = 1.0f; // distance to viewport
    float FOV_y = 90.0f;
    float aspect = float(img_width)/img_height;
    float vp_height = 2*f*tan((FOV_y/2) * M_PI/180.0f);
    float vp_width = vp_height * aspect;

    Point3 camera(0, 0, -1);
 
    // Output file setup
    ofstream img("output_img.ppm");
    img << "P3\n" << img_width << " " << img_height << "\n" << max_col_val << "\n";
 
    // Random sphres generation
    const int num_spheres = 20;
    Sphere spheres[num_spheres];
    for(int i=0;i<num_spheres;i++){
        spheres[i].center.x = (randf()-0.5f)*vp_width;
        spheres[i].center.y = (randf()-0.5f)*vp_height;
        spheres[i].center.z = 1.0f + randf()*9.0f; // in front of camera
        spheres[i].radius = 0.1f + randf()*0.2f; // small radius
    }

    for(int j=0;j<img_height;j++){
        for(int i=0;i<img_width;i++){
            float u = float(i)/(img_width-1);
            float v = float(j)/(img_height-1);

            float x_pix = (u-0.5f)*vp_width;
            float y_pix = (0.5f-v)*vp_height;

            Point3 ray_dir(x_pix - camera.x, y_pix - camera.y, 0 - camera.z);
            Ray ray(camera, ray_dir);

            float closest_t = 1e9;
            for(int k=0;k<num_spheres;k++){
                float t = intersectSphere(ray, spheres[k]);
                if(t < closest_t) closest_t = t;
            }

            // Depth shading
            float val = 0;
            if(closest_t < 1e9){
                float max_depth = 10.0f;
                val = 255 * (1.0f - closest_t / max_depth);
                val = max(0.0f, min(255.0f, val));
            }

            int iv = int(val);
            img << iv << " " << iv << " " << iv << "\n";
        }
    }

    img.close();
    return 0;
}
