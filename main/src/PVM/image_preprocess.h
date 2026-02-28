#ifndef IMAGE_PREPROCESS_H
#define IMAGE_PREPROCESS_H

#include "hls_stream.h"
#include "types.h"

class ImagePreprocess {
public:
    // Local constructor prevents linker "undefined symbol" errors
    ImagePreprocess(int h, int w, int d) : H(h), W(w), D(d) {}
    
    void forward(const float *image, hls::stream<PixelVec> &out_stream);

private:
    int H, W, D;
};

#endif