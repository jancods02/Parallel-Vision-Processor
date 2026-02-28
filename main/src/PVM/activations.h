#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "types.h"
#include "hls_math.h"

static ssm_t exp_lut_approx(ssm_t val) {
    #pragma HLS INLINE

    static const ssm_t lut_values[18] = {
        1.0000, 0.6065, 0.3678, 0.2231,
        0.1353, 0.0820, 0.0497, 0.0301,
        0.0183, 0.0111, 0.0067, 0.0040,
        0.0024, 0.0015, 0.0009, 0.0005,
        0.0000, 0.0000 
    };
    #pragma HLS BIND_STORAGE variable=lut_values type=rom_1p impl=bram

    if (val < 0) return 1.0;
    if (val > 8) return 0.0;

    // FIX: Multiply explicitly by ssm_t to avoid type promotion warnings
    ssm_t scaled = val * ssm_t(2.0); 
    
    // FIX: Use a direct C++ cast instead of .to_int() to prevent ap_fixed negative shift warnings
    int idx = (int)scaled; 

    ssm_t frac = scaled - (ssm_t)idx;

    ssm_t y0 = lut_values[idx];
    ssm_t y1 = lut_values[idx+1];
   
    return y0 + frac * (y1 - y0);
}

static ssm_t softplus_approx(ssm_t x) {
    #pragma HLS INLINE
    if (x > 3) return x;
    if (x < -3) return 0;
   
    if (x < -1) return (ssm_t(0.13) * x) + ssm_t(0.43);
    if (x < 1)  return (ssm_t(0.50) * x) + ssm_t(0.69);
    return (ssm_t(0.87) * x) + ssm_t(0.32);
}

static ssm_t silu_approx(ssm_t x) {
    #pragma HLS INLINE
    ssm_t sig;
    if (x > 4) sig = 1;
    else if (x < -4) sig = 0;
    else sig = (ssm_t(0.5) * x) * ssm_t(0.25) + ssm_t(0.5);
   
    return x * sig;
}

#endif