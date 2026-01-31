#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include "FFmpegUtils.h"

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

/**
 * @brief Metadata header for the compressed video file format.
 */
struct VideoMetadata {
    int width;              ///< Video width in pixels.
    int height;             ///< Video height in pixels.
    int frameInterval;      ///< Step size for processing frames (e.g., every 10th frame).
    int64_t duration;       ///< Total duration in AV_TIME_BASE units.
    int totalFrames;        ///< Total number of sampled frames stored in the file.
    int keyframeInterval;   ///< Interval at which full keyframes are inserted.
    int changeThreshold;    ///< Sensitivity threshold for delta changes (0-255).
};

// -----------------------------------------------------------------------------
// Video Compressor Class
// -----------------------------------------------------------------------------

/**
 * @class VideoCompressor
 * @brief Handles the full pipeline of opening, decoding, compressing, and saving video data.
 */
class VideoCompressor {
private:
    std::string inputFile;
    std::string outputFile;
    int frameInterval;
    int keyframeInterval;
    int changeThreshold;
    int targetWidth;
    int targetHeight;
    int quantization; // Bit-shift factor (0-4) default 0
    bool useCuda;

    int totalProcessedFrames = 0;
    int savedFrameCount = 0;
    
    // Statistics for final reporting
    size_t keyframeCount = 0;
    size_t totalKeyframeBytes = 0;
    size_t totalDeltaBytes = 0;

    // FFmpeg Resources
    AVFormatContextPtr fmtCtx;
    AVCodecContextPtr codecCtx;
    SwsContextPtr swsCtx;
    AVBufferRefPtr hwDeviceCtx;
    int videoStreamIdx = -1;

    // File I/O
    VideoMetadata metadata{};
    std::vector<uint8_t> lastFrame; // Holds the previously processed frame for delta calculation
    std::ofstream ofs;
    std::streampos headerPos;       // File position to rewrite header metadata at the end

    // State Tracking for Reuse of SWS Context
    int currentSwsW = -1;
    int currentSwsH = -1;
    enum AVPixelFormat currentSwsFormat = AV_PIX_FMT_NONE;

    // Error State
    bool cudaErrorOccurred = false;

public:
    /**
     * @brief Construct a new Video Compressor.
     * 
     * @param input Path to input video file.
     * @param output Path to output compressed file.
     * @param interval Frame step interval (e.g. process every Nth frame).
     * @param kfInterval Keyframe interval (every N saved frames).
     * @param threshold Delta threshold (difference required to register a pixel change).
     * @param targetW Target width (0/negative for input width).
     * @param targetH Target height (0/negative for input height).
     * @param quantize Bit-shift quantization (0-4) to reduce noise and improve runs.
     * @param cuda Whether to attempt CUDA hardware acceleration.
     */
    VideoCompressor(const std::string& input, const std::string& output,
        int interval, int kfInterval, int threshold, 
        int targetW, int targetH, int quantize, bool cuda);

    ~VideoCompressor();

    /**
     * @brief Initializes the FFmpeg context, codecs, and output file.
     * 
     * @return true if initialization is successful, false otherwise.
     */
    bool initialize();

    /**
     * @brief Executes the main compression loop.
     * 
     * Reads frames from input, decodes (SW or HW), downloads if necessary, converts to YUV420P,
     * compresses, and streams to disk.
     * 
     * @return true on success, false on failure (or if fallback is needed).
     */
    bool compress();

    /**
     * @brief Checks if a CUDA error occurred during processing.
     * @return true if an unrecoverable CUDA error happened.
     */
    bool hasCudaError() const;

private:
    /**
     * @brief Writes the initial file header with placeholder values.
     */
    void writeHeader();

    /**
     * @brief Finalizes the file by writing the correct metadata and frame counts.
     */
    void finalizeFile();

    /**
     * @brief Analyzes a YUV420P frame and compresses it.
     * 
     * @param frame The AVFrame containing YUV420P data.
     * @param pts Presentation Timestamp.
     */
    void processAndWriteFrame(AVFrame* frame, int64_t pts);

    /**
     * @brief Writes a single compressed frame packet to disk.
     */
    void writeFrameToDisk(int64_t pts, uint8_t frameType, const std::vector<uint8_t>& data);

    /**
     * @brief Compresses data using Run-Length Encoding (RLE).
     */
    std::vector<uint8_t> compressKeyframe(const std::vector<uint8_t>& data);

    /**
     * @brief Compresses data using a threshold-based Delta algorithm.
     */
    std::vector<uint8_t> compressDelta(const std::vector<uint8_t>& current, const std::vector<uint8_t>& previous);

    /**
     * @brief Writes a variable-length integer (VarInt) to the buffer.
     */
    void writeVarint(std::vector<uint8_t>& out, size_t value);

    double MP4FileSize(const std::string& filePath);

    void printStats();
};
