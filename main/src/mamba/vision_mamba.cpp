#include "vision_mamba.h"

VisionMambaBlock::VisionMambaBlock(int h, int w, int d) 
    : H(h), W(w), D(d),
      norm(d), 
      in_proj(d), 
      conv(d), 
      param_gen(d), 
      ssm(d), 
      out_block(d) 
{}

void VisionMambaBlock::run(
    hls::stream<PixelVec> &input_stream,
    hls::stream<PixelVec> &output_stream
) {
    #pragma HLS DATAFLOW

    int L = H * W;

    static hls::stream<PixelVec> s_residual_copy("s_res");
    static hls::stream<PixelVec> s_norm_out("s_norm");
    static hls::stream<PixelVec> s_main_branch("s_main");
    static hls::stream<PixelVec> s_gate_branch("s_gate");
    static hls::stream<PixelVec> s_conv_out("s_conv");
    static hls::stream<S6Params> s_params("s_params");
    static hls::stream<PixelVec> s_ssm_out("s_ssm");

    // FIFO Depths
    #pragma HLS STREAM variable=s_residual_copy depth=1024 // Needs to store data while others process
    #pragma HLS STREAM variable=s_gate_branch   depth=1024 // Delay match for Gate
    #pragma HLS STREAM variable=s_norm_out      depth=4
    #pragma HLS STREAM variable=s_main_branch   depth=4
    #pragma HLS STREAM variable=s_conv_out      depth=4
    #pragma HLS STREAM variable=s_params        depth=4
    #pragma HLS STREAM variable=s_ssm_out       depth=4

    static hls::stream<PixelVec> s_input_to_norm("s_in_norm");
    #pragma HLS STREAM variable=s_input_to_norm depth=4
    
    for(int t=0; t<L; t++) {
        #pragma HLS PIPELINE II=1
        PixelVec p = input_stream.read();
        s_residual_copy.write(p); 
        s_input_to_norm.write(p); 
    }

    
    // x -> RMSNorm
    norm.forward(L, s_input_to_norm, s_norm_out);

    // RMSNorm -> (Main, Gate)
    in_proj.forward(L, s_norm_out, s_main_branch, s_gate_branch);

    // Main -> Conv1D
    conv.forward(L, s_main_branch, s_conv_out);

    // Conv1D -> Params
    param_gen.forward(L, s_conv_out, s_params);

    // Params -> SSM Core
    ssm.forward(L, s_params, s_ssm_out);

    // (SSM, Gate, Residual) -> Output
    out_block.forward(L, s_ssm_out, s_gate_branch, s_residual_copy, output_stream);
}