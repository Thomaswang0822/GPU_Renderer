#pragma once

// CMake insert NDEBUG when building with RelWithDebInfo
// This is an ugly hack to undo that...
#undef NDEBUG

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>

// for suppressing unused warnings
#define UNUSED(x) (void)(x)

// We use double for most of our computation.
// Rendering is usually done in single precision Reals.
// However, torrey is an educational renderer with does not
// put emphasis on the absolute performance. 
// We choose double so that we do not need to worry about
// numerical accuracy as much when we render.
// Switching to floating point computation is easy --
// just set Real = float.
using Real = float;

// Lots of PIs!
__device__ const Real c_PI = Real(3.14159265358979323846);
__device__ const Real c_INVPI = Real(1.0) / c_PI;
__device__ const Real c_TWOPI = Real(2.0) * c_PI;
__device__ const Real c_INVTWOPI = Real(1.0) / c_TWOPI;
__device__ const Real c_FOURPI = Real(4.0) * c_PI;
__device__ const Real c_INVFOURPI = Real(1.0) / c_FOURPI;
__device__ const Real c_PIOVERTWO = Real(0.5) * c_PI;
__device__ const Real c_PIOVERFOUR = Real(0.25) * c_PI;

__device__ const unsigned int MAX_DEPTH = 50;    // maximum recursion depth
__device__ const Real EPSILON = 1e-7;

__device__ const float c_INFINITY = std::numeric_limits<float>::infinity();

namespace fs = std::filesystem;

inline std::string to_lowercase(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::tolower);
    return out;
}

__device__ inline int modulo(int a, int b) {
    auto r = a % b;
    return (r < 0) ? r+b : r;
}

__device__ inline float modulo(float a, float b) {
    float r = ::fmodf(a, b);
    return (r < 0.0f) ? r+b : r;
}

__device__ inline double modulo(double a, double b) {
    double r = ::fmod(a, b);
    return (r < 0.0) ? r+b : r;
}

template <typename T>
__device__ inline T max(const T &a, const T &b) {
    return a > b ? a : b;
}

template <typename T>
__device__ inline T min(const T &a, const T &b) {
    return a < b ? a : b;
}

__device__ inline Real radians(const Real deg) {
    return (c_PI / Real(180)) * deg;
}

__device__ inline Real degrees(const Real rad) {
    return (Real(180) / c_PI) * rad;
}
