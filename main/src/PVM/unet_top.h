#ifndef UNET_TOP_H
#define UNET_TOP_H

#include "types.h"

// AXI mapped IP core signature
void unet_pvm_top(
    ssm_t *image_in,   // Input image [H * W * C]
    ssm_t *mask_out,   // Output mask [H * W * C]
    ssm_t *weights     // Flattened array for projection weights/biases
);

#endif