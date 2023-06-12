#ifndef BVH_H
#define BVH_H

#include "hittable.h"
#include "world.h"
#include "aabb.h"
#include "ray.h"
#include <thrust/sort.h>

class bvh_node : public hittable {
    public:
        __device__ bvh_node();

        __device__ bvh_node(world& w, float time0, float time1) : bvh_node((hittable**)w.list, int(0), int(w.list_size), time0, time1) {}

        __device__ bvh_node(
            hittable **l, int start, int end, float time0, float time1);

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(
            float t0, float t1, aabb& box) const override;

    public:
        hittable *left;
        hittable *right;
        aabb b;

};

__device__ int random_int(int min, int max, curandState *local_rand_state) {
    return min + int((max - min) * curand_uniform(local_rand_state));
}

__device__ inline bool box_compare(const hittable *a, const hittable *b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
        printf("No bounding box in bvh_node constructor.\n");

    return box_a.min.get(axis) < box_b.min.get(axis);
}

__device__ bool box_x_compare(const hittable *a, const hittable *b) {
    return box_compare(a, b, 0);
}

__device__ bool box_y_compare(const hittable *a, const hittable *b) {
    return box_compare(a, b, 1);
}

__device__ bool box_z_compare(const hittable *a, const hittable *b) {
    return box_compare(a, b, 2);
}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& box) const {
    box = b;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!b.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ bvh_node::bvh_node(hittable **l, int start, int end, float time0, float time1) {
    auto l2 = l;
    // generate a random state
    curandState s;
    curandState* local_rand_state = threadIdx.x == 0 ? &s : &s;
    int axis = random_int(0, 2, local_rand_state);
    auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                    : box_z_compare;
    int object_span = end - start;

    if (object_span == 1) {
        left = right = l2[start];
    } else if (object_span == 2) {
        if (comparator(l2[start], l2[start+1])) {
            left = l2[start];
            right = l2[start+1];
        } else {
            left = l2[start+1];
            right = l2[start];
        }
    } else {
        thrust::sort(thrust::seq, l2 + start, l2 + end, comparator);
        auto mid = start + object_span/2;
        left = new bvh_node(l2, start, mid, time0, time1);
        right = new bvh_node(l2, mid, end, time0, time1);
    }

    aabb box_left, box_right;

    if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
        printf("No bounding box in bvh_node constructor.\n");

    b = surrounding_box(box_left, box_right);
}




#endif