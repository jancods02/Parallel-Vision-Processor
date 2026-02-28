#ifndef VISION_MAMBA_H
#define VISION_MAMBA_H

#include "hls_stream.h"
#include "types.h"

class VisionMambaBlock {
public:
    int H, W, D;
    VisionMambaBlock(int h, int w, int d) : H(h), W(w), D(d) {}

    void run(
        hls::stream<PixelVec> &input_stream,
        hls::stream<PixelVec> &output_stream
    );
};

#endif