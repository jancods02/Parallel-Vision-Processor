#ifndef VISION_MAMBA_H
#define VISION_MAMBA_H

#include "hls_stream.h"
#include "types.h"
#include "layers.h"
#include "s6_layer.h"
#include "s6_param_gen.h"

class VisionMambaBlock {
public:
    int H, W, D;
    
    // Components
    RMSNorm norm;
    InputProjection in_proj;
    Conv1DBlock conv;
    S6ParamGen param_gen;
    S6Layer ssm;
    OutputBlock out_block;
    Splitter splitter;

    VisionMambaBlock(int h, int w, int d);

    void run(
        hls::stream<PixelVec> &input_stream,
        hls::stream<PixelVec> &output_stream
    );
};

#endif