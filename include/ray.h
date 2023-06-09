#pragma once

#include "utils.h"
#include "vector.h"

// From RTOW, ray.h
struct ray
{
    __device__ ray() {}
    __device__ ray(const Vector3 &origin, const Vector3 &direction)
        : orig(origin), dir(normalize(direction)) {}
    // 3rd Constructor: use orig and target point
    __device__ ray(const Vector3 &origin, const Vector3 &target, bool useTarget)
        : orig(origin)
    {this->dir = normalize(target - origin);}

    __device__ Vector3 at(float t) const{return orig + t * dir;}

    // ata
    int src = -1; // Debug: which sphere (surface) it originates from
    Vector3 orig;
    Vector3 dir;
};

__device__ inline ray mirror_ray(ray &rayIn, Vector3 outNormal, Vector3 &hitPt)
{
    if (dot(rayIn.dir, outNormal) > 0.0)
    {

        return mirror_ray(rayIn, -outNormal, hitPt);
    }
    Vector3 outDir = normalize(rayIn.dir - 2.f * dot(rayIn.dir, outNormal) * outNormal);
    return ray(
        hitPt, // origin
        outDir // reflected dir
    );
}

__device__ inline Vector3 mirror_dir(const Vector3 &in_dir, Vector3 normal_dir)
{
    if (dot(in_dir, normal_dir) < 0.0f)
    {
        return mirror_dir(in_dir, -normal_dir);
    }

    return normalize(2.f * dot(in_dir, normal_dir) * normal_dir - in_dir);
}

/**
 * @brief Given required info, generate a refract ray
 *
 * @note: this function works for both ray inside & outside sphere and
 * does not distinguish them.
 *
 * @param rayIn
 * @param outNormal
 * @param hitPt
 * @param eta_ratio: eta / eta_prime, which are refractive indices
 *
 * @return ray
 */
__device__ inline ray refract_ray(ray &rayIn, Vector3 outNormal, Vector3 &hitPt, float eta_ratio)
{
    float cos_theta = dot(-rayIn.dir, outNormal);
    Vector3 r_out_perp = eta_ratio * (rayIn.dir + cos_theta * outNormal);
    Vector3 r_out_parallel = -sqrt(fabs(1.0f - length_squared(r_out_perp))) * outNormal;
    return ray(
        hitPt,
        r_out_perp + r_out_parallel // constructor will take care of normalize()
    );
}

// Use Schlick's approximation for reflectance.
__device__ inline float reflectance(float cosine, float eta_ratio)
{
    auto r0 = (1.f - eta_ratio) / (1.f + eta_ratio);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * pow((1.f - cosine), 5);
}
