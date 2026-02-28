#include "s6_param_gen.h"
#include "activations.h"

S6ParamGen::S6ParamGen(int d) : D(d) {}

void S6ParamGen::forward(
    int L,
    hls::stream<PixelVec> &in_stream,
    hls::stream<S6Params> &out_stream
) {
    for (int t = 0; t < L; t++) {

        PixelVec p = in_stream.read();
        S6Params params;
        #pragma HLS ARRAY_PARTITION variable=params.delta complete
        #pragma HLS ARRAY_PARTITION variable=params.B complete
        #pragma HLS ARRAY_PARTITION variable=params.C complete
        #pragma HLS ARRAY_PARTITION variable=params.x complete

        for (int d = 0; d < 32; d++) {
// OPTIMIZATION: Throttle unroll factor
#pragma HLS PIPELINE II=1
            if (d < D) {
                // OPTIMIZATION: Keep everything in fixed-point to avoid float conversion hardware overhead
                ssm_t val = ssm_t(0.1) * p.data[d];
                
                params.delta[d] = softplus_approx(val);
                params.B[d]     = val;
                params.C[d]     = val;
                params.x[d]     = p.data[d];
            } else {
                params.delta[d] = 0; params.B[d] = 0; params.C[d] = 0; params.x[d] = 0;
            }
        }
        out_stream.write(params);
    }
}