#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "top.h"

#define H 32
#define W 32
#define D 3

// --- Helper: Load PPM Image ---
bool load_ppm(const char *filename, std::vector<float> &buffer, std::ofstream &log) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        log << "[WARNING] Could not open " << filename << ". Using dummy data." << std::endl;
        return false;
    }
    std::string type;
    int w, h, max_val;
    file >> type >> w >> h >> max_val;
    file.get(); // skip newline
    
    buffer.resize(H * W * D);
    if (type == "P6") {
        std::vector<unsigned char> temp(H * W * D);
        file.read(reinterpret_cast<char*>(temp.data()), temp.size());
        for (size_t i = 0; i < temp.size(); i++) buffer[i] = temp[i] / 255.0f;
    } else {
        int val;
        for (int i = 0; i < H * W * D; i++) { file >> val; buffer[i] = val / 255.0f; }
    }
    log << "[INFO] Loaded real image: " << filename << std::endl;
    return true;
}

// --- Helper: Save PPM ---
void save_ppm(const char *filename, const std::vector<float> &buffer) {
    std::ofstream file(filename);
    file << "P3\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < H * W * D; i++) {
        int val = (int)(buffer[i] * 255.0f);
        if (val < 0) val = 0; if (val > 255) val = 255;
        file << val << " ";
        if ((i + 1) % D == 0) file << "\n";
    }
}

int main() {
    std::ofstream log("simulation_log.txt");
    if (!log.is_open()) return 1;

    std::vector<float> image(H * W * D);
    std::vector<float> output(H * W * D);

    // Load Data
    if (!load_ppm("C:/RP-FPGA/input.ppm", image, log)) {
        for(int i=0; i<H*W*D; i++) image[i] = (float)(i % 255) / 255.0f;
    }

    // Run Hardware
    log << "[INFO] Running vim_top..." << std::endl;
    vim_top(H, W, D, image.data(), output.data());

    // Save & Verify
    save_ppm("output_processed_new.ppm", output);
    
    float max_val = 0;
    for(float f : output) if(f > max_val) max_val = f;

    if(max_val == 0) {
        log << "[FAIL] Output is all zeros." << std::endl;
        return 1;
    } else {
        log << "[PASS] Non-zero output generated." << std::endl;
        return 0;
    }
} 