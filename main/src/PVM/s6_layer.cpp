#include "s6_layer.h"
#include "activations.h"

S6Layer::S6Layer(int d) : D(d) {}

void S6Layer::forward(
    int L,
    hls::stream<S6Params> &in_stream,
    hls::stream<PixelVec> &out_stream
) {
    // Persistent state
    static ssm_t state[32];
    #pragma HLS ARRAY_PARTITION variable=state complete

    // Reset State at start of frame
    for (int d = 0; d < 32; d++) {
        #pragma HLS UNROLL
        state[d] = 0;
    }

    for (int t = 0; t < L; t++) {

#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024 avg=1024
        S6Params p = in_stream.read();
        PixelVec out_vec;
        #pragma HLS ARRAY_PARTITION variable=out_vec.data complete

        for (int d = 0; d < 32; d++) {
// OPTIMIZATION: Throttle unroll factor to 8 to prevent combinatorial explosion on exp_lut_approx
#pragma HLS PIPELINE II=1
            if (d < D) {
                ssm_t dt = p.delta[d];
                ssm_t b  = p.B[d];
                ssm_t c  = p.C[d];
                ssm_t x  = p.x[d];

                ssm_t decay = exp_lut_approx(dt);
               
                ssm_t current_state = state[d];
                // SSM Recurrence: h' = A*h + B*x
                ssm_t next_state = decay * current_state + dt * b * x;
               
                state[d] = next_state;
                
                // Output: y = C * h
                out_vec.data[d] = c * next_state;
            } else {
                out_vec.data[d] = 0;
            }
        }
        out_stream.write(out_vec);
    }
}