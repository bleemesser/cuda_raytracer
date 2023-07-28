#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"
#include "aabb.h"

class triangle : public hittable
{
public:
    __device__ __host__ triangle() {}
    __device__ __host__ triangle(vec3 A, vec3 B, vec3 C, material *mat_ptr) : A(A), B(B), C(C), mat_ptr(mat_ptr){};

    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &box) const;
    vec3 A, B, C;
    material *mat_ptr;
};

__device__ bool triangle::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    vec3 edge1 = B - A;
    vec3 edge2 = C - A;
    vec3 h = r.direction().cross(edge2);
    double a = edge1.dot(h);
    if (a > -0.00001 && a < 0.00001)
        return false;
    double f = 1.0 / a;
    vec3 s = r.origin() - A;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0)
        return false;
    vec3 q = s.cross(edge1);
    double v = f * r.direction().dot(q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    double t = f * edge2.dot(q);
    if (t > t_min && t < t_max)
    {
        rec.p = r.point_at(t);
        rec.t = t;
        rec.mat_ptr = mat_ptr;
        rec.normal = edge1.cross(edge2).normalize();
        return true;
    }
    return false;
}

__device__ bool triangle::bounding_box(float t0, float t1, aabb &output_box) const
{
    vec3 min = vec3(fmin(A.x(), fmin(B.x(), C.x())), fmin(A.y(), fmin(B.y(), C.y())), fmin(A.z(), fmin(B.z(), C.z())));
    vec3 max = vec3(fmax(A.x(), fmax(B.x(), C.x())), fmax(A.y(), fmax(B.y(), C.y())), fmax(A.z(), fmax(B.z(), C.z())));
    output_box = aabb(min, max);
    return true;
}

#endif