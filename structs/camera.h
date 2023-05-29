#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "vec3.h"
#include "curand_kernel.h"

class camera
{
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, float aperture)
    {
        float theta = vfov * 3.14159265358979323846 / 180;
        float half_height = tan(theta / 2);
        float height = 2 * half_height;
        float width = aspect * height;

        vec3 w = (lookfrom - lookat).normalize();
        vec3 u = vup.cross(w).normalize();
        vec3 v = w.cross(u);

        origin = lookfrom;
        horizontal = width * u * focus_dist;
        vertical = height * v * focus_dist;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;
        lens_radius = aperture / 2;
    }
    __device__ ray get_ray(float u, float v, curandState *local_rand_state)
    {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    float lens_radius;
    float focus_dist;
};

#endif