#include "top.h"
#include "image_preprocess.h"
#include "vision_mamba.h"
#include "hls_stream.h"
#include <string.h>

// Process 1: Hardware-Aware Input Mover
// Optimized for burst reading and initial patch embedding with position injection
void input_proc(int H, int W, int D, const float *image, hls::stream<PixelVec> &out_s) {
    #pragma HLS INLINE off
    ImagePreprocess input_converter(H, W, D);
    // forward() now handles float-to-fixed casting and position embedding injection
    input_converter.forward(image, out_s);
}

// Process 2: Bidirectional Compute Engine
// Orchestrates the parallel forward and backward S6 recurrence paths
void mamba_proc(int H, int W, int D, hls::stream<PixelVec> &in_s, hls::stream<PixelVec> &out_s) {
    #pragma HLS INLINE off
    VisionMambaBlock vim_block(H, W, D);
    // run() executes the bidirectional dataflow block
    vim_block.run(in_s, out_s);
}

// Process 3: Optimized Write-Back with Burst Support
// Ensures high-bandwidth write-out of segmentation masks/features to DDR
void write_back_burst(int H, int W, int D, hls::stream<PixelVec> &in, float *out) {
    #pragma HLS INLINE off
    float buffer[32];
    #pragma HLS ARRAY_PARTITION variable=buffer complete
    int L = H * W; 

    for(int t = 0; t < L; t++) {
        #pragma HLS PIPELINE II=1
        PixelVec v = in.read();
        
        for(int d = 0; d < 32; d++) {
            #pragma HLS UNROLL
            // Cast back to float only at the very boundary to save LUTs in core logic
            buffer[d] = (d < D) ? (float)v.data[d] : 0.0f;
        }
        // memcpy triggers AXI burst mode, essential for avoiding II=32 bottleneck
        memcpy(out + (t * D), buffer, D * sizeof(float));
    }
}

void vim_top(int H, int W, int D, const float *image, float *output) {
    // Port configurations for high-performance memory mapping
    #pragma HLS INTERFACE m_axi port=image  offset=slave bundle=gmem0 depth=3072 \
        max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=3072 \
        max_write_burst_length=256 num_write_outstanding=16
    
    #pragma HLS INTERFACE s_axilite port=H
    #pragma HLS INTERFACE s_axilite port=W
    #pragma HLS INTERFACE s_axilite port=D
    #pragma HLS INTERFACE s_axilite port=return

    // Global streams linking the processes. 
    // Static ensures persistence, depth prevents backpressure during bidirectional processing.
    static hls::stream<PixelVec> stream_in("stream_in");
    static hls::stream<PixelVec> stream_out("stream_out");
    #pragma HLS STREAM variable=stream_in  depth=512
    #pragma HLS STREAM variable=stream_out depth=512

    // DATAFLOW enables task-level parallelism (Overlapping Input/Compute/Output)
    #pragma HLS DATAFLOW
    input_proc(H, W, D, image, stream_in);
    mamba_proc(H, W, D, stream_in, stream_out);
    write_back_burst(H, W, D, stream_out, output);
}