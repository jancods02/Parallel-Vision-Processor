#ifndef LAYERS_H
#define LAYERS_H


#include "types.h"
#include "hls_math.h"
#include "activations.h"
#include "hls_stream.h"


// --- Class 1: RMS Normalization ---
class RMSNorm {
    int D;
    ssm_t weights[32];


public:
    RMSNorm(int d) : D(d) {
        for(int i=0; i<32; i++) weights[i] = 1.0;
    }


    void forward(
        int L,
        hls::stream<PixelVec> &in_stream,
        hls::stream<PixelVec> &out_stream
    ) {
        for(int t=0; t<L; t++) {
#pragma HLS PIPELINE II=1
            PixelVec in_vec = in_stream.read();
            PixelVec out_vec;


            ssm_t sum_sq = 0;
            for(int d=0; d<32; d++) {
#pragma HLS UNROLL
                if(d < D) sum_sq += in_vec.data[d] * in_vec.data[d];
            }
           
            ssm_t rsqrt = ssm_t(1.0) / hls::sqrt(sum_sq / ssm_t(D) + ssm_t(0.0001));


            for(int d=0; d<32; d++) {
#pragma HLS UNROLL
                if(d < D) out_vec.data[d] = in_vec.data[d] * rsqrt * weights[d];
                else      out_vec.data[d] = 0;
            }
            out_stream.write(out_vec);
        }
    }
};


// --- Class 2: Input Projection ---
class InputProjection {
    int D;
public:
    InputProjection(int d) : D(d) {}


    void forward(
        int L,
        hls::stream<PixelVec> &in_stream,
        hls::stream<PixelVec> &main_branch,
        hls::stream<PixelVec> &gate_branch
    ) {
        for(int t=0; t<L; t++) {
#pragma HLS PIPELINE II=1
            PixelVec x = in_stream.read();
            main_branch.write(x);
            gate_branch.write(x);
        }
    }
};


// --- Class 3: Causal Convolution ---
class Conv1DBlock {
    int D;
    ssm_t line_buffer[2][32];
    ssm_t weights[3][32];


public:
    Conv1DBlock(int d) : D(d) {
        for(int r=0; r<2; r++)
            for(int i=0; i<32; i++) line_buffer[r][i] = 0;
        for(int k=0; k<3; k++)
            for(int i=0; i<32; i++) weights[k][i] = 0.33;
    }


    void forward(
        int L,
        hls::stream<PixelVec> &in_stream,
        hls::stream<PixelVec> &out_stream
    ) {
        for(int t=0; t<L; t++) {
#pragma HLS PIPELINE II=1
            PixelVec in_vec = in_stream.read();
            PixelVec out_vec;


            for(int d=0; d<32; d++) {
#pragma HLS UNROLL
                if(d < D) {
                    ssm_t conv_val = in_vec.data[d] * weights[0][d] +
                                     line_buffer[0][d] * weights[1][d] +
                                     line_buffer[1][d] * weights[2][d];
                   
                    line_buffer[1][d] = line_buffer[0][d];
                    line_buffer[0][d] = in_vec.data[d];


                    out_vec.data[d] = silu_approx(conv_val);
                }
            }
            out_stream.write(out_vec);
        }
    }
};


// --- Class 4: Output Block ---
class OutputBlock {
    int D;
public:
    OutputBlock(int d) : D(d) {}


    void forward(
        int L,
        hls::stream<PixelVec> &ssm_stream,
        hls::stream<PixelVec> &gate_stream,
        hls::stream<PixelVec> &residual_stream,
        hls::stream<PixelVec> &final_out
    ) {
        for(int t=0; t<L; t++) {
#pragma HLS PIPELINE II=1
            PixelVec s = ssm_stream.read();
            PixelVec g = gate_stream.read();
            PixelVec r = residual_stream.read();
            PixelVec y;


            for(int d=0; d<32; d++) {
#pragma HLS UNROLL
                if(d < D) {
                    ssm_t gate_act = silu_approx(g.data[d]);
                    ssm_t fused = s.data[d] * gate_act;
                    y.data[d] = fused + r.data[d];
                }
            }
            final_out.write(y);
        }
    }
};


// --- Class 5: Splitter (NEW) ---
// Encapsulates the input splitting logic to satisfy Dataflow requirements
class Splitter {
public:
    void forward(
        int L,
        hls::stream<PixelVec> &in_stream,
        hls::stream<PixelVec> &to_norm,
        hls::stream<PixelVec> &to_residual
    ) {
        for(int t=0; t<L; t++) {
#pragma HLS PIPELINE II=1
            PixelVec p = in_stream.read();
            to_norm.write(p);
            to_residual.write(p);
        }
    }
};


#endif
