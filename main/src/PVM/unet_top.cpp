#include "unet_top.h"
#include "pvm_config.h"
#include "pvm_layer.h"

void unet_pvm_top(
    ssm_t *image_in, 
    ssm_t *mask_out, 
    ssm_t *weights
) {
    // FIX: Optimized depths for 4x4 resolution
    // image_in: 4*4 (H*W) * 32 (c_in) = 512
    // mask_out: 4*4 (H*W) * 64 (c_out) = 1024
    // weights: 64*32 (weights) + 64 (bias) = 2112
    #pragma HLS INTERFACE m_axi port=image_in bundle=gmem0 depth=512
    #pragma HLS INTERFACE m_axi port=mask_out bundle=gmem1 depth=1024
    #pragma HLS INTERFACE m_axi port=weights bundle=gmem2 depth=2112
    #pragma HLS INTERFACE s_axilite port=return

    // Static buffers sized based on config_enc5 (H=4, W=4, c_in=32, c_out=64)
    static ssm_t enc4_out[config_enc5::seq_len * config_enc5::c_in];
    static ssm_t enc5_out[config_enc5::seq_len * config_enc5::c_out];
    #pragma HLS ARRAY_PARTITION variable=enc5_out cyclic factor=4 dim=1

    // Map weights and calculate the bias offset
    const int weight_offset = config_enc5::c_out * config_enc5::c_in;
    const ssm_t *enc5_proj_w = weights; 
    const ssm_t *enc5_proj_b = weights + weight_offset;

    // Copy input to internal buffer with pipeline to aid burst inference
    for(int i=0; i < config_enc5::seq_len * config_enc5::c_in; i++) {
        #pragma HLS PIPELINE II=1
        enc4_out[i] = image_in[i];
    }

    // Execute PVMLayer Enc 5 [cite: 7]
    custom_pvm_layer<config_enc5>(
        enc4_out, 
        enc5_out, 
        enc5_proj_w, 
        enc5_proj_b
    );

    // Copy to output
    for(int i=0; i < config_enc5::seq_len * config_enc5::c_out; i++) {
        #pragma HLS PIPELINE II=1
        mask_out[i] = enc5_out[i];
    }
}