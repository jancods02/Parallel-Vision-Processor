#ifndef TYPES_H
#define TYPES_H

#include <ap_fixed.h>

typedef ap_fixed<18, 8, AP_RND, AP_SAT> ssm_t;

struct PixelVec {
    ssm_t data[32]; 
};

struct S6Params {
    ssm_t delta[32];
    ssm_t B[32];
    ssm_t C[32];
    ssm_t x[32];
};

#endif