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

        __device__ bvh_node(hittable** l, int num_hittables, float time0, float time1, int* num_new_hittables, curandState *local_rand_state)
            : bvh_node(l, 0, num_hittables, time0, time1, num_new_hittables, local_rand_state) {}


        __device__ bvh_node(
            hittable **l, int start, int end, float time0, float time1, int* num_new_hittables, curandState *local_rand_state);

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(
            float t0, float t1, aabb& box) const override;

    public:
        hittable *left;
        hittable *right;
        aabb l1;
        aabb r1;
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

__device__ bvh_node::bvh_node(hittable **l, int start, int end, float time0, float time1, int *num_new_hittables, curandState *local_rand_state) {
    printf("bvh: made it to constructor\n");
    int axis = random_int(0, 2, local_rand_state);
    auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                    : box_z_compare;
    printf("bvh: made it past comparator\n");
    int object_span = end - start;
    printf("object_span: %d\n", object_span);
    printf("start: %d\n", start);
    printf("end: %d\n", end);
    if (object_span == 1) {
        printf("Object span == 1\n");
        left = right = l[start];
        printf("bvh: made it past object_span == 1\n");
    } else if (object_span == 2) {
        printf("Object span == 2\n");
        if (comparator(l[start], l[start + 1])) {
            left = l[start];
            right = l[start + 1];
        } else {
            left = l[start + 1];
            right = l[start];
        }
        printf("bvh: made it past object_span == 2\n");
    } else {
        printf("Object span > 2\n");
        // thrust::sort(l + start, l + end, comparator);
        printf("bvh: made it past sort\n");
        int mid = start + object_span / 2;
        left = new bvh_node(l, start, mid, time0, time1, num_new_hittables, local_rand_state);
        right = new bvh_node(l, mid, end, time0, time1, num_new_hittables, local_rand_state);
        printf("bvh: made it past left and right\n");
        num_new_hittables += 2;
    }

    if (!left->bounding_box(time0, time1, l1)
        || !right->bounding_box(time0, time1, r1)
    )
        printf("No bounding box in bvh_node constructor.\n");
    printf("bvh: made it past bounding box\n");
    b = surrounding_box(l1, r1);
    printf("bvh: made it to end of constructor\n");
}




#endif