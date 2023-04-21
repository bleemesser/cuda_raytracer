#include <iostream>

// #include "structs/vec3.h"
// #include "structs/ray.h"

// #define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
// void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
// {
//     if (result)
//     {
//         std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
//         // Make sure we call CUDA Device Reset before exiting
//         cudaDeviceReset();
//         exit(99);
//     }
// }
// __device__ bool hit_sphere(const vec3 &center, float radius, const ray &r)
// {
//     vec3 oc = r.origin() - center;
//     float a = r.direction().dot(r.direction());
//     float b = 2.0f * oc.dot(r.direction());
//     float c = oc.dot(oc) - radius * radius;
//     float discriminant = b * b - 4.0f * a * c;
//     return (discriminant > 0.0f);
// }
// __device__ vec3 color(const ray &r)
// {
//     if (hit_sphere(vec3(0, 0, -1), 0.5, r))
//         return vec3(1, 0, 0);
//     vec3 unit_direction = r.direction().normalize();
//     float t = 0.5f * (unit_direction.y() + 1.0f);
//     return vec3(1.0, 1.0, 1.0) * (1.0f - t) + vec3(0.5, 0.7, 1.0) * t;
// }

// __global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin)
// {

//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int j = threadIdx.y + blockIdx.y * blockDim.y;

//     if ((i >= max_x) || (j >= max_y))
//         return;
//     int pixel_index = j * max_x + i;
//     float u = float(i) / float(max_x);
//     float v = float(j) / float(max_y);
//     ray r(origin, lower_left_corner + horizontal * u + vertical * v);
//     // fb[pixel_index] = color(r);
//     fb[pixel_index] = vec3(u, v, 0.2);
// }

int main()
{
    std::cout << "Hello World!" << std::endl;
    // const int nx = 1920;
    // const int ny = 1080;

    // int num_pixels = nx * ny;

    // size_t fb_size = num_pixels * sizeof(vec3);

    // vec3 *fb;

    // checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // int tx = 8;
    // int ty = 8;

    // dim3 blocks(nx / tx + 1, ny / ty + 1);
    // dim3 threads(tx, ty);

    // render<<<blocks, threads>>>(fb, nx, ny, vec3(-2.0, -1.0, -1.0), vec3(4.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 0.0));

    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    // // file output ppm
    // FILE *f = fopen("image.ppm", "w");
    // fprintf(f, "P3\n%d %d\n%d\n", nx, ny, 255);
    // for (int j = ny - 1; j >= 0; j--)
    // {
    //     for (int i = 0; i < nx; i++)
    //     {
    //         size_t pixel_index = j * nx + i;
    //         int ir = int(255.99 * fb[pixel_index].x());
    //         int ig = int(255.99 * fb[pixel_index].y());
    //         int ib = int(255.99 * fb[pixel_index].z());
    //         fprintf(f, "%d %d %d\n", ir, ig, ib);
    //     }
    // }
    // fclose(f);

    // checkCudaErrors(cudaFree(fb));

    return 0;
}