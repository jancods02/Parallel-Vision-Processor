#include "vision_mamba.h"
#include "layers.h"
#include "s6_layer.h"
#include "s6_param_gen.h"

// Helper function to act as a producer for the internal streams
void split_input(int L, hls::stream<PixelVec> &in, hls::stream<PixelVec> &out1, hls::stream<PixelVec> &out2) {
    for(int t=0; t<L; t++) {
        #pragma HLS PIPELINE II=1
        PixelVec p = in.read();
        out1.write(p);
        out2.write(p);
    }
}

void VisionMambaBlock::run(
    hls::stream<PixelVec> &input_stream,
    hls::stream<PixelVec> &output_stream
) {
    #pragma HLS INLINE off
    // Everything inside this region must be a function call or a stream declaration
    #pragma HLS DATAFLOW

    int L = H * W;

    // 1. Internal Stream Declarations (Non-static for instance isolation)
    hls::stream<PixelVec> s_res("s_res"), s_in_norm("s_in_norm"), s_norm_out("s_norm");
    hls::stream<PixelVec> s_main("s_main"), s_gate("s_gate"), s_conv_out("s_conv");
    hls::stream<S6Params> s_params_fwd("s_fwd");
    hls::stream<PixelVec> s_ssm_out("s_ssm");

    // 2. Set Depths (Crucial for the residual path to prevent deadlock)
    #pragma HLS STREAM variable=s_res depth=1024 
    #pragma HLS STREAM variable=s_gate depth=1024
    #pragma HLS STREAM variable=s_main depth=16
    #pragma HLS STREAM variable=s_in_norm depth=16
    #pragma HLS STREAM variable=s_norm_out depth=16
    #pragma HLS STREAM variable=s_ssm_out depth=16
    #pragma HLS STREAM variable=s_conv_out depth=16
    #pragma HLS STREAM variable=s_norm_out depth=16
    #pragma HLS STREAM variable=s_params_fwd depth=16
    // 3. Local instances of workers to ensure Resource Isolation
    RMSNorm norm_i(D);
    InputProjection in_proj_i(D);
    Conv1DBlock conv_i(D);
    S6ParamGen param_gen_i(D);
    S6Layer ssm_i(D);
    OutputBlock out_block_i(D);

    // 4. Dataflow Functional Pipeline
    split_input(L, input_stream, s_res, s_in_norm);
    
    norm_i.forward(L, s_in_norm, s_norm_out);
    in_proj_i.forward(L, s_norm_out, s_main, s_gate);
    conv_i.forward(L, s_main, s_conv_out);
    
    // Call matching your bidirectional definition in s6_param_gen.cpp
    param_gen_i.forward(L, s_conv_out, s_params_fwd);
    
    ssm_i.forward(L, s_params_fwd, s_ssm_out);
    
    out_block_i.forward(L, s_ssm_out, s_gate, s_res, output_stream);
}