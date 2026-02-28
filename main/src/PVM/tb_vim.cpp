#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include "unet_top.h"
#include "pvm_config.h"

// --- Helper: Generate Safe Dummy Weights ---
void fill_with_dummy_weights(std::vector<ssm_t>& arr) {
    for (size_t i = 0; i < arr.size(); i++) {
        // Generate small values between -0.05 and +0.05 to prevent fixed-point overflow
        float rand_val = ((float)rand() / RAND_MAX) * 0.1f - 0.05f; 
        arr[i] = (ssm_t)rand_val;
    }
}

// --- Helper: Load PPM Image ---
// Adapts a 3-channel RGB image into the input tensor (padding extra channels with 0)
bool load_ppm(const char *filename, std::vector<ssm_t> &buffer, int target_H, int target_W, int target_C) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[WARNING] Could not open " << filename << ". Check path!" << std::endl;
        return false;
    }
    std::string type;
    int w, h, max_val;
    file >> type >> w >> h >> max_val;
    file.get(); // skip newline
    
    if (w != target_W || h != target_H) {
        std::cerr << "[WARNING] Image dims (" << w << "x" << h << ") don't match target (" 
                  << target_W << "x" << target_H << ")." << std::endl;
    }

    std::vector<unsigned char> temp(target_H * target_W * 3, 0);
    if (type == "P6") {
        file.read(reinterpret_cast<char*>(temp.data()), temp.size());
    } else {
        int val;
        for (int i = 0; i < target_H * target_W * 3; i++) { 
            file >> val; 
            temp[i] = (unsigned char)val; 
        }
    }

    // Map the 3-channel image into the target channel size (e.g., 32 or 3)
    for (int t = 0; t < target_H * target_W; t++) {
        for (int c = 0; c < target_C; c++) {
            if (c < 3) {
                buffer[t * target_C + c] = (ssm_t)(temp[t * 3 + c] / 255.0f);
            } else {
                buffer[t * target_C + c] = (ssm_t)0.0f; // Pad unused channels
            }
        }
    }
    std::cout << "[INFO] Successfully loaded real image: " << filename << std::endl;
    return true;
}

// --- Helper: Save Output Feature Map to PPM ---
// Extracts the first 3 channels of the output tensor to visualize as RGB
void save_ppm(const char *filename, const std::vector<ssm_t> &buffer, int out_H, int out_W, int out_C) {
    std::ofstream file(filename);
    file << "P3\n" << out_W << " " << out_H << "\n255\n";
    
    for (int t = 0; t < out_H * out_W; t++) {
        for (int c = 0; c < 3; c++) {
            int val = 0;
            if (c < out_C) {
                // Scale feature map for visibility and clamp
                float f_val = (float)buffer[t * out_C + c] * 255.0f * 10.0f; 
                val = (int)f_val;
                if (val < 0) val = 0; 
                if (val > 255) val = 255;
            }
            file << val << " ";
        }
        file << "\n";
    }
    std::cout << "[INFO] Saved output visualization to: " << filename << std::endl;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "[INFO] Starting PVM U-Net HLS Testbench..." << std::endl;
    std::cout << "===========================================" << std::endl;

    srand(time(NULL));

    // 1. Define dimensions based on your active config
    int H = config_enc5::H;         
    int W = config_enc5::W;
    int seq_len = config_enc5::seq_len; 
    int c_in = config_enc5::c_in;       
    int c_out = config_enc5::c_out;     

    int image_size = seq_len * c_in;
    int mask_size = seq_len * c_out;
    int weights_size = (c_out * c_in) + c_out; 

    // 2. Allocate memory
    std::vector<ssm_t> image_in(image_size, (ssm_t)0);
    std::vector<ssm_t> mask_out(mask_size, (ssm_t)0);
    std::vector<ssm_t> weights(weights_size, (ssm_t)0);

    // 3. Load Real Image & Generate Dummy Weights
    // Make sure to put a small test image at this path, or update the path!
    if (!load_ppm("C:/RP-FPGA/input.ppm", image_in, H, W, c_in)) {
        std::cout << "[FAIL] Exiting due to missing input image." << std::endl;
        return 1;
    }
    
    fill_with_dummy_weights(weights);

    // 4. Execute the Hardware IP Core
    std::cout << "[INFO] Executing hardware module unet_pvm_top..." << std::endl;
    unet_pvm_top(image_in.data(), mask_out.data(), weights.data());
    std::cout << "[INFO] Hardware execution complete." << std::endl;

    // 5. Save the output
    save_ppm("output_feature_map.ppm", mask_out, H, W, c_out);

    // 6. Basic Sanity Check
    int zero_count = 0;
    float max_val = -9999.0f;

    for (int i = 0; i < mask_size; i++) {
        float val = (float)mask_out[i];
        if (val > max_val) max_val = val;
        if (val == 0.0f) zero_count++;
    }

    std::cout << "[RESULT] Output Max Value: " << max_val << std::endl;
    if (zero_count == mask_size) {
        std::cout << "[FAIL] Output is entirely zeros." << std::endl;
        return 1;
    }

    std::cout << "[PASS] Testbench completed successfully." << std::endl;
    return 0;
}