#ifndef WORLD_H
#define WORLD_H

#include "hittable.h"
#include "aabb.h"

class world: public hittable {
    public:
        __device__ world() {}
        __device__ world(hittable **l, int n) { list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
        hittable **list;
        int list_size;
};

__device__ bool world::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // // first check if the ray hits the bounding box
    // aabb box;
    // if (!bounding_box(t_min, t_max, box)) return false;
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(minf(box0.min.x(), box1.min.x()),
                 minf(box0.min.y(), box1.min.y()),
                 minf(box0.min.z(), box1.min.z()));

    vec3 big(maxf(box0.max.x(), box1.max.x()),
               maxf(box0.max.y(), box1.max.y()),
               maxf(box0.max.z(), box1.max.z()));

    return aabb(small,big);
}

__device__ bool world::bounding_box(float t0, float t1, aabb& box) const {
    if (list_size < 1) return false;
    aabb temp_box;
    bool first_box = true;

    for (int i = 0; i < list_size; i++) {
        if (!list[i]->bounding_box(t0, t1, temp_box)) return false;
        box = first_box ? temp_box : surrounding_box(box, temp_box);
        first_box = false;
    }
    return true;
}

#endif