#include <iostream>
#include <float.h>
#include "structs/vec3.h"
#include "structs/ray.h"
#include "structs/hittable.h"
#include "structs/world.h"
#include "structs/sphere.h"
#include "structs/camera.h"
#include "structs/material.h"
#include <curand_kernel.h>

// CUDA STUFF
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void rand_init(curandState *rand_state, uint seed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state);
    }
}

// COLOR
__device__ vec3 color(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = cur_ray.direction().normalize();
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

// RENDER INIT
__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// RENDER
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}
#define RND (curand_uniform(&local_rand_state))


// SETUP
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, int num_hittables, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-5000.0,0), 5000,
                               new matte(vec3(0.2, 0.2, 0.2)));
        int i = 1;
        int span = 5;
        for(int a = -span; a < span; a++) {
            for(int b = -span; b < span; b++) {
                float choose_mat = RND;
                vec3 center(a+0.9f*RND,0.2,b+0.9f*RND);
                if(choose_mat < 0.25f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new matte(vec3(RND*0.2f, RND*0.4f, RND)));
                }
                else if(choose_mat < 0.65f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(RND, RND, RND), 0.0f));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new transparent(1.5, vec3(RND*0.2f, RND*0.4f, RND)));
                }
            }
        }
        // d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new transparent(1.5));
        // d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new matte(vec3(0.4, 0.2, 0.1)));
        // d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new world(d_list, num_hittables);

        vec3 lookfrom(0,1,-15);
        vec3 lookat(0,0,0);
        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 0.04;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 8.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *d_camera;
}

int main()
{
    // IMAGE PARAMS
    const int nx = 2560;
    const float ratio = 16.0f / 9.0f;
    const int ny = int(nx / ratio);
    const int ns = 3000;
    const int tx = 8;
    const int ty = 8;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // ALLOCATE FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // ALLOCATE RANDOM STATE
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2, time(0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // ALLOCATE WORLD
    hittable **d_list;
    int num_hittables = 10 * 10 + 1; // 2*span*2*span + floor + any other objects
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hittables * sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, num_hittables, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // RENDER
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // file output ppm
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", nx, ny, 255);

    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].x());
            int ig = int(255.99 * fb[pixel_index].y());
            int ib = int(255.99 * fb[pixel_index].z());
            // validate that the color is in range, if it's similar to the negative 32 bit int, it's probably a NaN and should be averaged
            ir = ir > 255 ? 255 : ir;
            ir = ir < 0 ? 0 : ir;
            ig = ig > 255 ? 255 : ig;
            ig = ig < 0 ? 0 : ig;
            ib = ib > 255 ? 255 : ib;
            ib = ib < 0 ? 0 : ib;

            fprintf(f, "%d %d %d\n", ir, ig, ib);
        }
    }
    fclose(f);

    // CLEANUP
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
    return 0;
}