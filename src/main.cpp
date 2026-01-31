#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "VideoCompressor.h"

/**
 * @brief Application Entry Point
 */
void printHelp(const char* progName) {
    std::cout << "Video Delta Compressor (YUV420P Streaming)" << std::endl;
    std::cout << "Usage: " << progName << " <input.mp4> <output.ige> [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --cuda             Enable CUDA hardware acceleration (NVIDIA only)" << std::endl;
    std::cout << "  --width <W>        Resize video to target width" << std::endl;
    std::cout << "  --height <H>       Resize video to target height" << std::endl;
    std::cout << "  --in-width <W>     Input width (required for raw .yuv)" << std::endl;
    std::cout << "  --in-height <H>    Input height (required for raw .yuv)" << std::endl;
    std::cout << "  --quantize <0-4>   Bit-shift quantization for noise reduction (default: 0)" << std::endl;
    std::cout << "  --interval <N>     Process every Nth frame (default: 10)" << std::endl;
    std::cout << "  --keyframe <N>     Insert a keyframe every N saved frames (default: 30)" << std::endl;
    std::cout << "  --threshold <T>    Pixel change threshold (0-255) (default: 15)" << std::endl;
    std::cout << "  --help             Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << progName << " input.mp4 output.ige --width 640 --height 360 --quantize 2" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Start of program" << std::endl;

    bool useCuda = false;
    std::vector<std::string> args;
    int targetWidth = 0;
    int targetHeight = 0;
    int inputWidth = 0;
    int inputHeight = 0;
    int quantization = 0;
    int frameInterval = 10;
    int keyframeInterval = 48;
    int changeThreshold = 15;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        } else if (arg == "--cuda") {
            useCuda = true;
        } else if (arg == "--width" && i + 1 < argc) {
            targetWidth = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            targetHeight = std::stoi(argv[++i]);
        } else if (arg == "--in-width" && i + 1 < argc) {
            inputWidth = std::stoi(argv[++i]);
        } else if (arg == "--in-height" && i + 1 < argc) {
            inputHeight = std::stoi(argv[++i]);
        } else if (arg == "--quantize" && i + 1 < argc) {
            quantization = std::stoi(argv[++i]);
        } else if (arg == "--interval" && i + 1 < argc) {
            frameInterval = std::stoi(argv[++i]);
        } else if (arg == "--keyframe" && i + 1 < argc) {
            keyframeInterval = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            changeThreshold = std::stoi(argv[++i]);
        } else {
            args.push_back(arg);
        }
    }

    if (args.size() < 2) {
        std::cout << "Error: Missing input/output file arguments." << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    std::string inputFile = args[0];
    std::string outputFile = args[1];
    
    // Support legacy positional args if provided and flags weren't used (optional/override)
    if (args.size() >= 3) frameInterval = std::stoi(args[2]);
    if (args.size() >= 4) keyframeInterval = std::stoi(args[3]);
    if (args.size() >= 5) changeThreshold = std::stoi(args[4]);

    std::cout << "Video Delta Compressor" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input: " << inputFile << std::endl;
    std::cout << "Output: " << outputFile << std::endl;
    std::cout << "Frame interval: " << frameInterval << std::endl;
    std::cout << "Keyframe interval: " << keyframeInterval << std::endl;
    std::cout << "Change threshold: " << changeThreshold << std::endl;
    std::cout << "HW Acceleration: " << (useCuda ? "CUDA" : "None") << std::endl;
    if (targetWidth > 0 && targetHeight > 0) {
        std::cout << "Target Resolution: " << targetWidth << "x" << targetHeight << std::endl;
    }
    std::cout << "Quantization: " << quantization << " bits" << std::endl << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Robust Execution with Retry Mechanism
    // Attempts to run with requested settings. If CUDA fails, it automatically falls back
    // to software decoding to ensure the user gets a result.
    while (true) {
        VideoCompressor compressor(inputFile, outputFile, frameInterval, keyframeInterval, changeThreshold, 
            targetWidth, targetHeight, inputWidth, inputHeight, quantization, useCuda);

        if (!compressor.initialize()) {
            std::cerr << "Failed to initialize compressor" << std::endl;
            if (useCuda) {
                std::cout << "Retrying with Software Decoding..." << std::endl;
                useCuda = false;
                continue;
            }
            return 1;
        }

        if (!compressor.compress()) {
            std::cerr << "Compression failed" << std::endl;
            if (useCuda && compressor.hasCudaError()) {
                std::cout << "CUDA Error detected. Retrying with Software Decoding..." << std::endl;
                useCuda = false;
                continue;
            }
            return 1;
        }
        break; // Success
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = endTime - startTime;
    long long micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    if (micros < 1000) std::cout << "\nTime taken: " << micros << " µs" << std::endl;
    else if (micros < 1'000'000) std::cout << "\nTime taken: " << micros / 1000.0 << " ms" << std::endl;
    else std::cout << "\nTime taken: " << micros / 1'000'000.0 << " s" << std::endl;

    std::cout << "\nCompression completed successfully!" << std::endl;

    return 0;
}