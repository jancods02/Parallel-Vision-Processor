#ifndef PVM_LAYER_H
#define PVM_LAYER_H

#include "types.h"
#include "vision_mamba.h"
#include "hls_stream.h"
#include "hls_math.h"

// Sub-function 1: Read Array, LayerNorm, Split to 4 Streams
template<typename CONFIG_T>
void pvm_split_and_norm(
    ssm_t *data_in, 
    hls::stream<PixelVec> out_streams[4]
) {
    #pragma HLS INLINE off
    const int seq_len = CONFIG_T::seq_len;
    const int c_in = CONFIG_T::c_in;
    const int chunk_dim = CONFIG_T::chunk_dim;
    const ssm_t inv_c_in = (ssm_t)(1.0f / c_in);

    // REMOVED PIPELINE HERE: Prevents forced unrolling of everything inside
    for (int t = 0; t < seq_len; t++) {
        
        ssm_t x[128]; 
        // FIX: Completely partition to allow parallel operations without port conflicts
        #pragma HLS ARRAY_PARTITION variable=x complete
        
        ssm_t mean = 0;
        for (int c = 0; c < c_in; c++) {
            #pragma HLS PIPELINE II=1
            x[c] = data_in[t * c_in + c];
            mean += x[c];
        }
        mean = mean * inv_c_in;

        ssm_t var = 0;
        for (int c = 0; c < c_in; c++) {
            #pragma HLS PIPELINE II=1
            ssm_t diff = x[c] - mean;
            var += diff * diff;
        }
        var = var * inv_c_in;
        
        float temp_var = (float)(var + (ssm_t)1e-5);
        ssm_t rsqrt = (ssm_t)(1.0f / hls::sqrt(temp_var));

        // Split into 4 PixelVec streams
        for (int chunk = 0; chunk < 4; chunk++) {
            #pragma HLS PIPELINE II=1
            PixelVec vec;
            for (int d = 0; d < 32; d++) {
                #pragma HLS UNROLL
                if (d < chunk_dim) {
                    vec.data[d] = (x[(chunk * chunk_dim) + d] - mean) * rsqrt;
                } else {
                    vec.data[d] = 0;
                }
            }
            out_streams[chunk].write(vec);
        }
    }
}

// Sub-function 2: Merge Streams, Skip Connection, LayerNorm, and Project
template<typename CONFIG_T>
void pvm_merge_and_project(
    ssm_t *data_in, 
    hls::stream<PixelVec> in_streams[4], 
    ssm_t *data_out,
    const ssm_t *proj_weights,
    const ssm_t *proj_bias
) {
    #pragma HLS INLINE off
    const int seq_len = CONFIG_T::seq_len;
    const int c_in = CONFIG_T::c_in;
    const int c_out = CONFIG_T::c_out;
    const int chunk_dim = CONFIG_T::chunk_dim;
    const ssm_t inv_c_in = (ssm_t)(1.0f / c_in);
    const ssm_t skip_scale = (ssm_t)CONFIG_T::skip_scale_val;

    ssm_t local_proj_w[CONFIG_T::c_out][CONFIG_T::c_in];
    ssm_t local_proj_b[CONFIG_T::c_out];
    
    // FIX: Completely partition dimension 2 so the 64-channel inner loop can read all weights instantly
    #pragma HLS ARRAY_PARTITION variable=local_proj_w complete dim=2

    for (int out_c = 0; out_c < c_out; out_c++) {
        local_proj_b[out_c] = proj_bias[out_c];
        for (int in_c = 0; in_c < c_in; in_c++) {
            #pragma HLS PIPELINE II=1
            local_proj_w[out_c][in_c] = proj_weights[out_c * c_in + in_c];
        }
    }

    // REMOVED PIPELINE HERE: Prevents forced unrolling of the heavy matrix multiplication
    for (int t = 0; t < seq_len; t++) {

        ssm_t merged[128];
        // FIX: Completely partition
        #pragma HLS ARRAY_PARTITION variable=merged complete

        // Read 4 streams and apply skip scale
        for (int chunk = 0; chunk < 4; chunk++) {
            #pragma HLS PIPELINE II=1
            PixelVec vec = in_streams[chunk].read();
            for (int d = 0; d < 32; d++) {
                #pragma HLS UNROLL
                if (d < chunk_dim) {
                    int orig_idx = (chunk * chunk_dim) + d;
                    ssm_t orig_val = data_in[t * c_in + orig_idx];
                    merged[orig_idx] = vec.data[d] + (skip_scale * orig_val);
                }
            }
        }

        // Second LayerNorm
        ssm_t mean = 0;
        for (int c = 0; c < c_in; c++) {
            #pragma HLS PIPELINE II=1
            mean += merged[c];
        }
        mean = mean * inv_c_in;

        ssm_t var = 0;
        for (int c = 0; c < c_in; c++) {
            #pragma HLS PIPELINE II=1
            ssm_t diff = merged[c] - mean;
            var += diff * diff;
        }
        var = var * inv_c_in;
        
        float temp_var = (float)(var + (ssm_t)1e-5);
        ssm_t rsqrt = (ssm_t)(1.0f / hls::sqrt(temp_var));

        ssm_t norm_merged[128];
        // FIX: Completely partition so the projection loop below has access to all elements
        #pragma HLS ARRAY_PARTITION variable=norm_merged complete
        
        for (int c = 0; c < c_in; c++) {
            #pragma HLS PIPELINE II=1
            norm_merged[c] = (merged[c] - mean) * rsqrt;
        }

        // Linear Projection Output
        // The outer loop handles 1 output channel per clock cycle. The inner loop gets completely unrolled.
        for (int out_c = 0; out_c < c_out; out_c++) {
            #pragma HLS PIPELINE II=1
            ssm_t out_val = local_proj_b[out_c]; 
            for (int in_c = 0; in_c < c_in; in_c++) {
                // Since local_proj_w and norm_merged are completely partitioned, 
                // this 64x unroll synthesizes cleanly and instantly.
                out_val += norm_merged[in_c] * local_proj_w[out_c][in_c]; 
            }
            data_out[t * c_out + out_c] = out_val;
        }
    }
}

// Top-Level PVM Layer (DATAFLOW Region)
template<typename CONFIG_T>
void custom_pvm_layer(
    ssm_t *data_in,
    ssm_t *data_out,
    const ssm_t *proj_weights,
    const ssm_t *proj_bias
) {
    #pragma HLS DATAFLOW

    hls::stream<PixelVec> mamba_in[4];
    hls::stream<PixelVec> mamba_out[4];
    #pragma HLS STREAM variable=mamba_in depth=16
    #pragma HLS STREAM variable=mamba_out depth=16

    pvm_split_and_norm<CONFIG_T>(data_in, mamba_in);

    VisionMambaBlock mamba_block_0(CONFIG_T::H, CONFIG_T::W, CONFIG_T::chunk_dim);
    VisionMambaBlock mamba_block_1(CONFIG_T::H, CONFIG_T::W, CONFIG_T::chunk_dim);
    VisionMambaBlock mamba_block_2(CONFIG_T::H, CONFIG_T::W, CONFIG_T::chunk_dim);
    VisionMambaBlock mamba_block_3(CONFIG_T::H, CONFIG_T::W, CONFIG_T::chunk_dim);

    mamba_block_0.run(mamba_in[0], mamba_out[0]);
    mamba_block_1.run(mamba_in[1], mamba_out[1]);
    mamba_block_2.run(mamba_in[2], mamba_out[2]);
    mamba_block_3.run(mamba_in[3], mamba_out[3]);

    pvm_merge_and_project<CONFIG_T>(data_in, mamba_out, data_out, proj_weights, proj_bias);
}

#endif