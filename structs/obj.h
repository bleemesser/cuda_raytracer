#ifndef OBJ_H
#define OBJ_H

// load obj file to get triangles, return list of triangles

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class triangle; // triangle(vec3 A, vec3 B, vec3 C, material *mat_ptr) : A(A), B(B), C(C), mat_ptr(mat_ptr){}
class world;

std::vector<triangle*> load_obj(std::string filename) {
    std::vector<triangle*> triangles;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::index_t> indices;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    for (const auto& shape : shapes) {
        const auto& mesh = shape.mesh;
        const auto& indices = mesh.indices;

        for (size_t i = 0; i < indices.size(); i += 3) {
            const auto& idx0 = indices[i + 0].vertex_index;
            const auto& idx1 = indices[i + 1].vertex_index;
            const auto& idx2 = indices[i + 2].vertex_index;

            const float* p0 = &attrib.vertices[3 * idx0];
            const float* p1 = &attrib.vertices[3 * idx1];
            const float* p2 = &attrib.vertices[3 * idx2];

            triangles.push_back(new triangle(vec3(p0[0], p0[1], p0[2]),
                                             vec3(p1[0], p1[1], p1[2]),
                                             vec3(p2[0], p2[1], p2[2]),
                                             nullptr));
        }
    }

    return triangles;
}

#endif