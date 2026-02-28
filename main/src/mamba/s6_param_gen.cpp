#include "s6_param_gen.h"


static float softplus_approx(float x) {
#pragma HLS INLINE
   


    if (x > 3.0f) {
        return x;
    }


    else if (x < -3.0f) {
        return 0.0f;
    }


    else {
 
        if (x < -1.0f) {
            return (0.13f * x) + 0.43f;
        }
 
        else if (x < 1.0f) {
            return (0.50f * x) + 0.69f;
        }


        else {
            return (0.87f * x) + 0.32f;
        }
    }
}


S6ParamGen::S6ParamGen(int d) : D(d) {}


void S6ParamGen::forward(
    int L,
    hls::stream<PixelVec> &in_stream,
    hls::stream<S6Params> &out_stream
) {
    for (int t = 0; t < L; t++) {
#pragma HLS PIPELINE II=1
        PixelVec p = in_stream.read();
        S6Params params;
        #pragma HLS ARRAY_PARTITION variable=params.delta complete
        #pragma HLS ARRAY_PARTITION variable=params.B complete
        #pragma HLS ARRAY_PARTITION variable=params.C complete
        #pragma HLS ARRAY_PARTITION variable=params.x complete


        for (int d = 0; d < 32; d++) {
#pragma HLS UNROLL
            if (d < D) {
                // FIXED: Explicitly cast the constant to the fixed-point type
                float val = (float)((ssm_t)0.1 * p.data[d]);
               
                params.delta[d] = softplus_approx(val);
               
                params.B[d]     = val;
                params.C[d]     = val;
                params.x[d]     = p.data[d];
            }
        }
        out_stream.write(params);
    }
}
