#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"
#include <curand_kernel.h>
struct hit_record;

__device__ float reflectance(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted)
{
    vec3 uv = v.normalize();
    float dt = uv.dot(n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
    {
        return false;
    }
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0f * v.dot(n) * n;
}

class material
{
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;
    __device__ virtual vec3 emit(float u, float v, const vec3 &p) const
    {
        return vec3(0, 0, 0);
    }
};

class matte : public material
{
public:
    __device__ matte(const vec3 &a) : albedo(a) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

public:
    vec3 albedo;
};

class metal : public material
{
public:
    __device__ metal(const vec3 &a, float f) : albedo(a)
    {
        if (f < 1)
            roughness = f;
        else
            roughness = 1;
    }
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 reflected = reflect(r_in.direction().normalize(), rec.normal);
        scattered = ray(rec.p, reflected + roughness * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (scattered.direction().dot(rec.normal) > 0.0f);
    }


public:
    vec3 albedo;
    float roughness;
};

class transparent : public material
{
public:
    __device__ transparent(float ri, vec3 albedo) : ref_idx(ri), albedo(albedo) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = albedo;
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (r_in.direction().dot(rec.normal) > 0.0f)
        {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = ref_idx * r_in.direction().dot(rec.normal) / r_in.direction().length();
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -r_in.direction().dot(rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        {
            reflect_prob = reflectance(cosine, ref_idx);
        }
        else
        {
            scattered = ray(rec.p, reflected);
            reflect_prob = 1.0f;
        }
        if (curand_uniform(local_rand_state) < reflect_prob)
        {
            scattered = ray(rec.p, reflected);
        }
        else
        {
            scattered = ray(rec.p, refracted);
        }
        return true;
    }

public:
    float ref_idx;
    vec3 albedo;
};

class light : public material
{
    public:
    vec3 color;
    float intensity;
    __device__ light(vec3 color, float intensity) : color(color), intensity(intensity) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        return false;
    }
    __device__ virtual vec3 emit(float u, float v, const vec3 &p) const
    {
        return color * intensity;
    }

};
#endif