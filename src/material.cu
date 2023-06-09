#include "material.h"


__device__ Real Material::compute_G1(const Vector3& h, const Vector3& dir, Hit_Record& rec)
{
    Real G;
    switch (matType)
    {
    case MicrofacetBP:
        if (dot(dir, h) <= 0.f) { 
            G = 0.f; 
        } else {
            Real cos_theta = dot(dir, rec.normal);
            // cot = 1 / tan = cos / sin
            Real cotan_theta = cos_theta / sqrtf(1.f - cos_theta * cos_theta);  
            Real a = sqrtf(0.5f * exponent + 1.f) * cotan_theta;

            if (a >= 1.6f) { 
                G = 1.f; 
            } else {
                G = (3.535f * a + 2.181f * a * a) / (1.f + 2.276f * a + 2.577f * a * a);
            }
        }
        break;
    default:
        G = 0.f;
        break;
    }
    return G;
}


__device__ Vector3 Material::get_F(const Vector3& h, const Vector3& out_dir, Hit_Record& rec)
{
    Vector3 F(1.f, 1.f, 1.f);

    Vector3 Ks = eval_RGB(reflectance, rec.u, rec.v);;
    Real cos_theta = dot(rec.normal, out_dir);
    switch (matType)
    {
    case Mirror:
        Vector3 F0 = Ks;
        F = F0 + (-F0 + 1.f) * powf(1.f - cos_theta, 5);

    case Plastic:
        Real f0 = powf( (exponent - 1.f) / (exponent + 1.f), 2);
        Real f = f0 + (-f0 + 1.f) * pow(1.f - cos_theta, 5);
        F *= f;
        break;
    case BlinnPhong:
        F = Ks + (1.f - Ks) * powf(1.f - dot(h, out_dir), 5);
        break;
    case MicrofacetBP:
        F = Ks + (1.f - Ks) * powf(1.f - dot(h, out_dir), 5);
        break;
    default:
        break;
    }
    return F;
}


__device__ Real Material::compute_PDF(const Vector3& h, const Vector3& out_dir, Hit_Record& rec)
{
    Real pdf = 1.f;
    Real cosTerm = dot(rec.normal, out_dir);
    switch (matType)
    {
    case Diffuse:
        pdf = max(cosTerm, 0.f) * c_INVPI;
        break;
    case Mirror:
        break;
    case Plastic:
        Vector3 F = get_F(h, out_dir, rec);
        pdf = cosTerm * c_INVPI * (1.f - F.x);
        break;
    case Phong:
        pdf = (exponent + 1.f) * c_INVTWOPI * powf(cosTerm, exponent);
        break;
    case BlinnPhong:
        pdf = (exponent + 1.f) * powf(dot(rec.normal, h), exponent) *
            c_INVTWOPI / (4.f * dot(out_dir, h));
        break;
    case MicrofacetBP:
        pdf = (exponent + 1.f) * powf(dot(rec.normal, h), exponent) *
            c_INVTWOPI / (4.f * dot(out_dir, h));
        break;
    default:
        break;
    }
    return pdf;
}


__device__ Vector3 Material::compute_BRDF(const Vector3& h, const Vector3& in_dir,
        const Vector3& out_dir, Hit_Record& rec)
{
    Vector3 brdf(0.f, 0.f, 0.f);

    Vector3 F = get_F(h, out_dir, rec);
    Real cosTerm = dot(rec.normal, out_dir);
    Vector3 Kd = eval_RGB(reflectance, rec.u, rec.v);

    switch (matType)
    {
    case Diffuse:
        brdf = Kd * max(cosTerm, 0.f) * c_INVPI;
        break;
    case Mirror:
        break;
    case Plastic:
        brdf = Kd * max(cosTerm, 0.f) * c_INVPI * (1.f-F);
        break;
    case Phong:
        brdf = Kd * (exponent + 1.f) * c_INVTWOPI * powf(max(cosTerm, 0.f), exponent);
        break;
    case BlinnPhong:
        Real c = (exponent + 2.f) * c_INVFOURPI / (2.f - pow(2.f, -exponent/2));
        brdf = c * F * powf(dot(rec.normal, h), exponent);
        break;
    case MicrofacetBP:
        Real G = compute_G1(h, in_dir, rec) * compute_G1(h, out_dir, rec);
        // D(h) = C * dot(shading noromal, h)
        Real D = (exponent + 2.f) * c_INVTWOPI * powf(dot(rec.normal, h), exponent);
        // put together
        return F * D * G / (4.f * dot(rec.normal, in_dir));
        break;
    default:
        break;
    }
    return brdf;
}