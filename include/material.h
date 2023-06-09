#pragma once

#include "parse_scene.h"
#include "Hit_Record.h"
#include "utils.h"

using namespace std;

// GPU Update: cannot use variant: use polymorphism
/**
 * 1. get rid of using Color = std::variant<Vector3, ImageTexture>;
 * 2. turn all materials into a subclass of a base
 */

/**
 * @brief One MatColor maps to a Vector3 or one RGB img
 * 
 * @note: image data is written when the
 * constructor is called.
 * 
 */
struct MatColor {
    bool isImg;
    // for a simple RGB
    Vector3 rgb;
    // for an image texture
    Image3 img3;
    Real uscale = 1, vscale = 1;
    Real uoffset = 0, voffset = 0;
    __device__ MatColor() {}
    __device__ MatColor(const Vector3 _color) :
        rgb(_color) {}
    __device__ MatColor(const ParsedImageTexture& pImgTex) :
        isImg(true), 
        uscale(pImgTex.uscale), vscale(pImgTex.vscale),
        uoffset(pImgTex.uoffset), voffset(pImgTex.voffset) 
    {
        // copy the image from ParsedImageTexture
        int length = pImgTex.img3.height * pImgTex.img3.width;
        cudaMalloc(&img3.data, length * sizeof(Vector3));
        cudaMemcpy(img3.data, pImgTex.img3.data, length * sizeof(Vector3), cudaMemcpyHostToDevice);
    }

    __device__ ~MatColor() {
        if (isImg) {
            cudaFree(img3.data);
        }
    }


    /**
     * @brief Given a uv position, return a Vector3 RGB on the texture image
     * We do bilinear interpolation with the weighted mean approach.
     * @ref https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean
     * 
     */
    __device__ Vector3 blerp_color(Real u, Real v) const {
        // null guardian
        if (!isImg) {
            // TODO: change to scene background color
            return Vector3(0.5, 0.5, 0.5);
        }
        
        // convert
        Real x = img3.width  * modulo(uscale * u + uoffset, 1.0) - 1e-9;
        Real y = img3.height * modulo(vscale * v + voffset, 1.0) - 1e-9;

        // obtain x1 y1 x2 y2
        int x1 = static_cast<int>(x); int y1 = static_cast<int>(y);
        int x2 = x1 + 1; int y2 = y1 + 1;
        // do bilinear interpolation, Weighted Mean approach
        // (x2 - x1) * (y2 - y1) = 1
        Real w11 = (x2-x) * (y2-y);
        Real w12 = (x2-x) * (y-y1);
        Real w21 = (x-x1) * (y2-y);
        Real w22 = (x-x1) * (y-y1);
        // wrap around
        return  w11 * img3(x1, y1) + 
                w12 * img3(x1, y2%img3.height) + 
                w21 * img3(x2%img3.width, y1) + 
                w22 * img3(x2%img3.width, y2%img3.height);

    }
};


/**
 * @brief Given a MatColor, return a Vector3 RGB
 * if MatColor is Vector3, then trivial case
 * if MatColor is ImageTexture, call funciton
 * 
 * @param refl 
 * @param u,v uv coordinate of hit record
 * @return Vector3 
 */
__device__ inline Vector3 eval_RGB(const MatColor& refl, const Real u, const Real v) {
    if (!refl.isImg) {
        return refl.rgb;
    } 
    else {
        return refl.blerp_color(u, v);
    }
}

enum MaterialType {
    None,
    Diffuse,
    Mirror,
    Plastic,
    Phong,
    BlinnPhong,
    MicrofacetBP
};


class Material {
public:
    __device__ Material(MaterialType _tp, MatColor _color, Real _exp=1.0)
        : matType(_tp), reflectance(_color), exponent(_exp) {}
    __device__ ~Material() {}  // all primitives, default


    // data
    MaterialType matType;
    MatColor reflectance;
    Real exponent; // alpha, or index of refraction for Plastic


    // computation functions
    __device__ Real compute_PDF(const Vector3& h, const Vector3& out_dir, Hit_Record& rec);
    __device__ Vector3 compute_BRDF(const Vector3& h, const Vector3& in_dir,
            const Vector3& out_dir, Hit_Record& rec);
    __device__ Vector3 get_F(const Vector3& h, const Vector3& out_dir, Hit_Record& rec);
    // Microfacet use only
    __device__ Real compute_G1(const Vector3& h, const Vector3& dir, Hit_Record& rec);
    

};
