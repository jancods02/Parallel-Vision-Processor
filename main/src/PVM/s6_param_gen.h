#ifndef S6_PARAM_GEN_H
#define S6_PARAM_GEN_H

#include "hls_stream.h"
#include "types.h"

class S6ParamGen {
public:
    S6ParamGen(int d);
    void forward(
        int L,
        hls::stream<PixelVec> &in_stream,
        hls::stream<S6Params> &out_stream
    );
private:
    int D;
};

#endif