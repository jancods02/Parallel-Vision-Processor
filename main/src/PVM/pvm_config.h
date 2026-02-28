#ifndef PVM_CONFIG_H
#define PVM_CONFIG_H

#include "types.h"

struct config_enc5{
    static const int H = 4;
    static const int W = 4;
    static const int seq_len =  H * W;
    static const int c_in = 32;
    static const int c_out = 64;
    static const int chunk_dim = c_in / 4; // 8 channels per Mamba chunk
    
    static constexpr float skip_scale_val = 1.0f;
};

struct config_dec2 {
    static const int H = 1;
    static const int W = 1;
    static const int seq_len = H * W; // 1 token
    
    static const int c_in = 64;
    static const int c_out = 32;
    static const int chunk_dim = c_in / 4; // 16 channels per Mamba chunk
    
    static constexpr float skip_scale_val = 1.0f; 
};


#endif

