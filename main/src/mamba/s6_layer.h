#ifndef S6_LAYER_H
#define S6_LAYER_H

#include "hls_stream.h"
#include "types.h"

class S6Layer {
public:
    S6Layer(int d);
    void forward(
        int L,
        hls::stream<S6Params> &in_stream,
        hls::stream<PixelVec> &out_stream
    );
private:
    int D;
};

#endif