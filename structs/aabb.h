#ifndef AABB_H
#define AABB_H

#include "vec3.h"
#include "ray.h"

__host__ __device__ inline float minf(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float maxf(float a, float b) { return a > b ? a : b; }

class aabb {
    public:
        __device__ aabb() {}
        __device__ aabb(const vec3& a, const vec3& b) { min = a; max = b; }

        // vec3 min() const { return min; }
        // vec3 max() const { return max; }

        __device__ inline bool aabb::hit(const ray& r, double t_min, double t_max) const {
            for (int a = 0; a < 3; a++) {
                auto invD = 1.0f / r.direction().get(a);
                auto t0 = (min.get(a) - r.origin().get(a)) * invD;
                auto t1 = (max.get(a) - r.origin().get(a)) * invD;
                if (invD < 0.0f) {
                    // swap t0 and t1
                    auto temp = t0;
                    t0 = t1;
                    t1 = temp;
                }
                t_min = t0 > t_min ? t0 : t_min;
                t_max = t1 < t_max ? t1 : t_max;
                if (t_max <= t_min)
                    return false;
            }
            return true;
        }


        vec3 min;
        vec3 max;
};

#endif