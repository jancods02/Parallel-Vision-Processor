#include "image_preprocess.h"
#include <string.h> 

void ImagePreprocess::forward(const float *image, hls::stream<PixelVec> &out_stream) {
    float local_buf[32];
    #pragma HLS ARRAY_PARTITION variable=local_buf complete

    int L = H * W;
    for (int t = 0; t < L; t++) {
        #pragma HLS PIPELINE II=1
        // Burst read copies a block from DDR to local BRAM
        // This prevents the AXI bus from locking up during DATAFLOW
        memcpy(local_buf, image + (t * D), D * sizeof(float));

        PixelVec vec;
        #pragma HLS ARRAY_PARTITION variable=vec.data complete
        for (int d = 0; d < 32; d++) {
            #pragma HLS UNROLL
            vec.data[d] = (d < D) ? (ssm_t)local_buf[d] : (ssm_t)0;
        }
        out_stream.write(vec);
    }
}