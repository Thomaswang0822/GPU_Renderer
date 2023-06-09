#pragma once

#include <curand_kernel.h>


__global__ void render_init(int max_x, int max_y, curandState *rand_state);