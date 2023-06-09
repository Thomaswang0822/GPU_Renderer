#pragma once

#include "vector.h"

#include <string>
#include <cstring>
#include <vector>

// In GPU implementation, Image should ONLY be used on host (CPU)
// After all pixel values are written to frame buffer vec3* fb,
// we just copy the value into the Image

/// A N-channel image stored in a contiguous vector
/// The storage format is HWC -- outer dimension is height
/// then width, then channels.
template<typename T>
struct Image {
    __host__ __device__ Image() {}
    __host__ __device__ Image(int w, int h) : width(w), height(h) {
        data = new T[w * h];
    }
    __host__ __device__ ~Image() {
        delete[] data;  // array doesn't auto deallocate like vector
    }

    __host__ __device__ T &operator()(int x) {
        return data[x];
    }

    __host__ __device__ const T &operator()(int x) const {
        return data[x];
    }

    __host__ __device__ T &operator()(int x, int y) {
        return data[y * width + x];
    }

    __host__ __device__ const T &operator()(int x, int y) const {
        return data[y * width + x];
    }

    int width;
    int height;
    T* data;
};

using Image1 = Image<Real>;
using Image3 = Image<Vector3>;

/// Read from an 1 channel image. If the image is not actually
/// single channel, the first channel is used.
/// Supported formats: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
__host__ __device__ Image1 imread1(const fs::path &filename);
/// Read from a 3 channels image. 
/// If the image only has 1 channel, we set all 3 channels to the same color.
/// If the image has more than 3 channels, we truncate it to 3.
/// Undefined behavior if the image has 2 channels (does that even happen?)
/// Supported formats: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
__host__ __device__ Image3 imread3(const fs::path &filename);

/// Save an image to a file.
/// Supported formats: PFM & exr
__host__ __device__ void imwrite(const fs::path &filename, const Image3 &image);
