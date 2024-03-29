#include <iostream>
#include <float.h>
#include "structs/vec3.h"
#include "structs/ray.h"
#include "structs/hittable.h"
#include "structs/world.h"
#include "structs/sphere.h"
#include "structs/triangle.h"
#include "structs/camera.h"
#include "structs/material.h"
#include "structs/aabb.h"
#include "structs/bvh.h"
#include "tiny_obj_loader.h"
#include "structs/obj.h"
#include <curand_kernel.h>
#include "CImg.h"
#include <chrono>
#include <unordered_map>
#pragma comment(lib, "windowscodecs.lib") // WINDOWS
#pragma comment(lib, "gdi32.lib")         // WINDOWS
#pragma comment(lib, "user32.lib")        // WINDOWS

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

__global__ void rand_init(curandState *rand_state, unsigned int seed)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(seed, 0, 0, rand_state);
    }
}

// COLOR
__device__ vec3 color(const ray &r, hittable **world, curandState *local_rand_state, bool bg_gradient)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    vec3 cur_light = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                cur_light *= rec.mat_ptr->emit(rec.u, rec.v, rec.p);
                return cur_attenuation * cur_light;
            }
        }
        else
        {

            vec3 unit_direction = cur_ray.direction().normalize();
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return bg_gradient ? cur_attenuation * c : vec3(0.0, 0.0, 0.0);
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
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
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hittable **world, curandState *rand_state, bool bg_gradient)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state, bg_gradient);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] += col;
}

#define RND (curand_uniform(&local_rand_state))

// // SETUP
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, int num_hittables, int span, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -10000.0, 0), 10000,
                               new matte(vec3(0.9, 0.9, 0.9)));
        int i = 1;
        for (int a = -span; a < span; a++)
        {
            for (int b = -span; b < span; b++)
            {
                float choose_mat = RND;
                vec3 center(a + 0.9f * RND, 0.2, b + 0.9f * RND);
                vec3 color = vec3(RND * RND, RND * RND, RND * RND);
                if (choose_mat < 0.15f)
                {
                    d_list[i++] = new sphere(center, 0.2f,
                                             new light(color, 6));
                }
                else if (choose_mat < 0.3f)
                {
                    d_list[i++] = new sphere(center, 0.2f,
                                             new matte(color));
                }
                else if (choose_mat < 0.5f)
                {
                    d_list[i++] = new sphere(center, 0.2f,
                                             new metal(color, 0.0f));
                }
                else
                {
                    d_list[i++] = new sphere(center, 0.2, new transparent(1.5, color));
                }
            }
        }
        d_list[i++] = new sphere(vec3(12, 25, 8), 10, new light(vec3(1.0, 1.0, 1.0), 15.0));
        d_list[i++] = new sphere(vec3(0, 1, 0), 1, new metal(vec3(1.0, 1.0, 1.0), 0.0));

        printf("Made it to w\n");
        *d_world = new world(d_list, num_hittables);

        vec3 lookfrom(0, 2, -15);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.04;
        *d_camera = new camera(lookfrom,
                               lookat,
                               vec3(0, 1, 0),
                               25.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

__global__ void construct_bvh(hittable **d_world, hittable **d_bvh, int num_hittables, curandState *rand_state)
{
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *d_camera;
}

// take in the cimg and a frame buffer and write the frame buffer to the display applying the edge effect
cimg_library::CImg<unsigned char> edge_effect(cimg_library::CImg<unsigned char> image, vec3 *fb, int f, int nx, int ny)
{
    // apply edge effect
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            int pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].x());
            int ig = int(255.99 * fb[pixel_index].y());
            int ib = int(255.99 * fb[pixel_index].z());

            // clamp values and write to cimg
            ir = ir > 255 ? 255 : ir;
            ir = ir < 0 ? 0 : ir;
            ig = ig > 255 ? 255 : ig;
            ig = ig < 0 ? 0 : ig;
            ib = ib > 255 ? 255 : ib;
            ib = ib < 0 ? 0 : ib;

            // apply edge effect
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
            {
                ir = 0;
                ig = 0;
                ib = 0;
            }
            else
            {
                int ir2 = int(255.99 * fb[pixel_index + 1].x());  // right
                int ig2 = int(255.99 * fb[pixel_index + 1].y());  // right
                int ib2 = int(255.99 * fb[pixel_index + 1].z());  // right
                int ir3 = int(255.99 * fb[pixel_index - 1].x());  // left
                int ig3 = int(255.99 * fb[pixel_index - 1].y());  // left
                int ib3 = int(255.99 * fb[pixel_index - 1].z());  // left
                int ir4 = int(255.99 * fb[pixel_index + nx].x()); // up
                int ig4 = int(255.99 * fb[pixel_index + nx].y()); // up
                int ib4 = int(255.99 * fb[pixel_index + nx].z()); // up
                int ir5 = int(255.99 * fb[pixel_index - nx].x()); // down
                int ig5 = int(255.99 * fb[pixel_index - nx].y()); // down
                int ib5 = int(255.99 * fb[pixel_index - nx].z()); // down
                ir = (ir + ir2 + ir3 + ir4 + ir5) / 5;
                ig = (ig + ig2 + ig3 + ig4 + ig5) / 5;
                ib = (ib + ib2 + ib3 + ib4 + ib5) / 5;
            }
            image(i, ny - j - 1, 0, 0) = ir / (f + 1);
            image(i, ny - j - 1, 0, 1) = ig / (f + 1);
            image(i, ny - j - 1, 0, 2) = ib / (f + 1);
        }
    }
    return image;
}

int main(int argc, char **argv)
{
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    // IMAGE PARAMS
    int nx = 640;
    const float ratio = 16.0f / 9.0f;
    int ny = int(nx / ratio);
    int ns = 1000;
    const int tx = 8;
    const int ty = 8;
    bool BG_GRADIENT = true;
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    int PROGRESS_DISPLAY_PERCENTAGE = 10;
    // grab command line arguments that are flagged -x, -ns, -bg (bool) -p (progress display percentage)
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-x") == 0)
        {
            i++;
            nx = atoi(argv[i]);
            ny = int(nx / ratio);
            num_pixels = nx * ny;
            fb_size = num_pixels * sizeof(vec3);
        }
        else if (strcmp(argv[i], "-ns") == 0)
        {
            i++;
            ns = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-bg") == 0)
        {
            i++;
            BG_GRADIENT = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            i++;
            PROGRESS_DISPLAY_PERCENTAGE = atoi(argv[i]);
        }
    }
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

    // // ALLOCATE WORLD
    hittable **d_list;
    int num_hittables = 10 * 10 + 1 + 1 + 1; // 2*span*2*span + floor + any other objects
    // int num_hittables = 1;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hittables * sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, num_hittables, 5, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    hittable **d_bvh;
    checkCudaErrors(cudaMalloc((void **)&d_bvh, num_hittables * sizeof(hittable *)));
    construct_bvh<<<1, 1>>>(d_list, d_bvh, num_hittables, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // RENDER
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // use cimg to create an empty display window
    cimg_library::CImg<unsigned char> cimg(nx, ny, 1, 3, 0);
    cimg_library::CImgDisplay display(cimg, "Ray Tracer");
    vec3 *temp_fb;
    vec3 *prev_frame;
    int prev_frame_index = 0;
    int display_state = 0;
    // normal malloc for temp_fb: we want to store the frame on the host when writing it out to avoid too many invdividual copy calls
    temp_fb = (vec3 *)malloc(fb_size);
    prev_frame = (vec3 *)malloc(fb_size);
    bool render_complete = false;
    for (int f = 0; f < ns; f++)
    {
        if (display.is_closed())
            break;
        if (display.is_key(49))
        {
            display_state = 0;
        }
        if (display.is_key(50) && f > 0)
        {
            display_state = 1;
        }
        if (display.is_key(51))
        {
            display_state = 2;
        }
        // if up arrow is pressed, increase progress display percentage
        if (display.is_keyARROWUP())
        {
            if (PROGRESS_DISPLAY_PERCENTAGE + 50 <= ns)
            {
                printf("Increasing progress display percentage to %d\n", PROGRESS_DISPLAY_PERCENTAGE + 50);
                PROGRESS_DISPLAY_PERCENTAGE += 50;
            }
        }
        // if down arrow is pressed, decrease progress display percentage
        if (display.is_keyARROWDOWN())
        {
            if (PROGRESS_DISPLAY_PERCENTAGE - 50 >= 1)
            {
                printf("Decreasing progress display percentage to %d\n", PROGRESS_DISPLAY_PERCENTAGE - 50);
                PROGRESS_DISPLAY_PERCENTAGE -= 50;
            }
        }
        // if ] is pressed, increase number of samples
        if (display.is_keyW())
        {
            printf("Increasing number of samples to %d\n", ns + 100);
            ns += 100;
        }
        // if [ is pressed, decrease number of samples
        if (display.is_keyS())
        {
            if (ns - 100 >= 1 && ns - 200 > f)
            {
                printf("Decreasing number of samples to %d\n", ns - 100);
                ns -= 100;
            }
        }
        printf("Rendering frame %d\n", f);
        render<<<blocks, threads>>>(fb, nx, ny, 1, d_camera, d_world, d_rand_state, BG_GRADIENT);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        if (f % int(ns / PROGRESS_DISPLAY_PERCENTAGE) == 0 || f == ns - 1)
        {
            // copy fb to temp_fb on the host
            checkCudaErrors(cudaMemcpy(temp_fb, fb, fb_size, cudaMemcpyDeviceToHost));
            if (display_state == 0)
            {
                printf("Displaying frame %d\n", f);
                // overwrite the previous cimg display with the new image
                for (int j = ny - 1; j >= 0; j--)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        int pixel_index = j * nx + i;
                        int ir = int(255.99 * temp_fb[pixel_index].x() / (f + 1));
                        int ig = int(255.99 * temp_fb[pixel_index].y() / (f + 1));
                        int ib = int(255.99 * temp_fb[pixel_index].z() / (f + 1));

                        // clamp values and write to cimg
                        cimg(i, ny - j - 1, 0, 0) = ir > 255 ? 255 : ir;
                        cimg(i, ny - j - 1, 0, 1) = ig > 255 ? 255 : ig;
                        cimg(i, ny - j - 1, 0, 2) = ib > 255 ? 255 : ib;
                    }
                }
                // display the image
                cimg.display(display);
            }
            else if (display_state == 1 && prev_frame != NULL) // if key 1 is pressed and the previous frame is not null, display the difference between the previous frame and the current frame
            {
                printf("Displaying difference between frame %d and frame %d\n", f, f - 1);
                // overwrite the previous cimg display with the new image
                for (int j = ny - 1; j >= 0; j--)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        int pixel_index = j * nx + i;
                        int ir = int(255.99 * (temp_fb[pixel_index].x() - prev_frame[pixel_index].x()) / ((f + 1) - prev_frame_index + 1));
                        int ig = int(255.99 * (temp_fb[pixel_index].y() - prev_frame[pixel_index].y()) / ((f + 1) - prev_frame_index + 1));
                        int ib = int(255.99 * (temp_fb[pixel_index].z() - prev_frame[pixel_index].z()) / ((f + 1) - prev_frame_index + 1));

                        // clamp values and write to cimg
                        cimg(i, ny - j - 1, 0, 0) = ir > 255 ? 255 : ir;
                        cimg(i, ny - j - 1, 0, 1) = ig > 255 ? 255 : ig;
                        cimg(i, ny - j - 1, 0, 2) = ib > 255 ? 255 : ib;
                    }
                }
                // display the image
                cimg.display(display);
            }
            else if (display_state == 2)
            {
                printf("Displaying edge effect for frame %d\n", f);
                cimg = edge_effect(cimg, temp_fb, f, nx, ny);
                cimg.display(display);
            }
            // copy temp_fb to prev_frame
            memcpy(prev_frame, temp_fb, fb_size);
            prev_frame_index = f;
        }
        if (f == ns - 1)
        {
            render_complete = true;
        }
        // if (display.is_keyR())
        // {
        //     create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, num_hittables, 5, d_rand_state2);
        //     checkCudaErrors(cudaGetLastError());
        //     checkCudaErrors(cudaDeviceSynchronize());
        //     f = 0;
        //     // clear fb
        //     checkCudaErrors(cudaMemset(fb, 0, fb_size));
        // }
    }

    // stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // output render time in minutes and seconds
    int minutes = duration.count() / 1000.0f / 60.0f;
    int seconds = duration.count() / 1000.0f - minutes * 60.0f;
    std::cout << "Render time: " << minutes << " minutes " << seconds << " seconds" << std::endl;

    cimg.display(display);

    // keep the window open until the user closes it
    while (!display.is_closed())
    {
        display.wait();
    }
    if (render_complete)
    {
        checkCudaErrors(cudaMemcpy(temp_fb, fb, fb_size, cudaMemcpyDeviceToHost));

        FILE *f = fopen("image.ppm", "w");
        fprintf(f, "P3\n%d %d\n%d\n", nx, ny, 255);

        for (int j = ny - 1; j >= 0; j--)
        {
            for (int i = 0; i < nx; i++)
            {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * temp_fb[pixel_index].x() / ns);
                int ig = int(255.99 * temp_fb[pixel_index].y() / ns);
                int ib = int(255.99 * temp_fb[pixel_index].z() / ns);
                // validate that the color is in range
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
    }

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
}