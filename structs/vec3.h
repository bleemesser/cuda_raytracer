#ifndef VEC3_H
#define VEC3_H

class vec3
{
public:
    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(float e0, float e1, float e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float &operator[](int i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &v2)
    {
        e[0] += v2.e[0];
        e[1] += v2.e[1];
        e[2] += v2.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const vec3 &v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator/=(const float t)
    {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const
    {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    // vector operations
    __host__ __device__ vec3 operator+(const vec3 &v) const { return vec3(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]); };
    __host__ __device__ vec3 operator-(const vec3 &v) const { return vec3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]); };
    __host__ __device__ vec3 operator*(const vec3 &v) const { return vec3(e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]); };
    __host__ __device__ vec3 operator/(const vec3 &v) const { return vec3(e[0] / v.e[0], e[1] / v.e[1], e[2] / v.e[2]); };

    // scalar operations (must be used vec3 * scalar, not scalar * vec3)
    __host__ __device__ vec3 operator*(float f) const { return vec3(e[0] * f, e[1] * f, e[2] * f); };
    __host__ __device__ vec3 operator/(float f) const { return vec3(e[0] / f, e[1] / f, e[2] / f); };
    __host__ __device__ vec3 operator-(float f) const { return vec3(e[0] - f, e[1] - f, e[2] - f); };
    __host__ __device__ vec3 operator+(float f) const { return vec3(e[0] + f, e[1] + f, e[2] + f); };

    // important math functions
    __host__ __device__ float dot(const vec3 &v) const { return e[0] * v.e[0] + e[1] * v.e[1] + e[2] * v.e[2]; };
    __host__ __device__ vec3 cross(const vec3 &v) const { return vec3(e[1] * v.e[2] - e[2] * v.e[1], e[2] * v.e[0] - e[0] * v.e[2], e[0] * v.e[1] - e[1] * v.e[0]); };
    __host__ __device__ vec3 normalize() const { return *this / length(); };

    // index operator
    __host__ __device__ float get(int i) const { return e[i]; };

public:
    float e[3];
};

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(float t, const vec3 &v)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}



#endif