#include <iostream>
#include <fstream>  
#include <vector>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <filesystem> 
#include <string>
#include <algorithm>
#include <map>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

struct CompressedFrame {
    int64_t timestamp;
    uint8_t frameType; // 0=delta, 1=keyframe
    std::vector<uint8_t> data; // Compressed data
};

struct VideoMetadata {
    int width;
    int height;
    int frameInterval;
    int64_t duration;
    int totalFrames;
    int keyframeInterval;
    int changeThreshold;
};

class VideoCompressor {
private:
    std::string inputFile;
    std::string outputFile;
    int frameInterval;
    int keyframeInterval;
    int changeThreshold;
    int totalProcessedFrames = 0;
    int savedFrameCount = 0;

    AVFormatContext* fmtCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    SwsContext* swsCtx = nullptr;
    int videoStreamIdx = -1;

    VideoMetadata metadata{};
    std::vector<CompressedFrame> compressedFrames;
    std::vector<uint8_t> lastFrame;

public:
    VideoCompressor(const std::string& input, const std::string& output,
        int interval = 10, int kfInterval = 30, int threshold = 15)
        : inputFile(input), outputFile(output), frameInterval(interval),
        keyframeInterval(kfInterval), changeThreshold(threshold) {
    }

    ~VideoCompressor() {
        cleanup();
    }

    bool initialize() {
        if (avformat_open_input(&fmtCtx, inputFile.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Could not open input file: " << inputFile << std::endl;
            return false;
        }

        if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
            std::cerr << "Could not find stream information" << std::endl;
            return false;
        }

        for (unsigned i = 0; i < fmtCtx->nb_streams; i++) {
            if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStreamIdx = i;
                break;
            }
        }

        if (videoStreamIdx == -1) {
            std::cerr << "Could not find video stream" << std::endl;
            return false;
        }

        AVCodecParameters* codecParams = fmtCtx->streams[videoStreamIdx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);

        if (!codec) {
            std::cerr << "Unsupported codec" << std::endl;
            return false;
        }

        codecCtx = avcodec_alloc_context3(codec);
        if (!codecCtx) {
            std::cerr << "Could not allocate codec context" << std::endl;
            return false;
        }

        if (avcodec_parameters_to_context(codecCtx, codecParams) < 0) {
            std::cerr << "Could not copy codec params to context" << std::endl;
            return false;
        }

        if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
            std::cerr << "Could not open codec" << std::endl;
            return false;
        }

        metadata.width = codecCtx->width;
        metadata.height = codecCtx->height;
        metadata.frameInterval = frameInterval;
        metadata.keyframeInterval = keyframeInterval;
        metadata.changeThreshold = changeThreshold;
        metadata.duration = fmtCtx->duration;
        metadata.totalFrames = 0;

        swsCtx = sws_getContext(
            codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
            codecCtx->width, codecCtx->height, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );

        if (!swsCtx) {
            std::cerr << "Could not initialize SWS context" << std::endl;
            return false;
        }

        return true;
    }

    bool compress() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        AVFrame* rgbFrame = av_frame_alloc();

        if (!packet || !frame || !rgbFrame) {
            std::cerr << "Could not allocate packet/frame" << std::endl;
            return false;
        }

        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codecCtx->width, codecCtx->height, 1);
        uint8_t* buffer = (uint8_t*)av_malloc((size_t)numBytes);
        if (!buffer) {
            std::cerr << "Could not allocate RGB buffer" << std::endl;
            av_frame_free(&rgbFrame);
            av_frame_free(&frame);
            av_packet_free(&packet);
            return false;
        }
        av_image_fill_arrays(rgbFrame->data, rgbFrame->linesize, buffer, AV_PIX_FMT_RGB24, codecCtx->width, codecCtx->height, 1);
        rgbFrame->width = codecCtx->width;
        rgbFrame->height = codecCtx->height;
        rgbFrame->format = AV_PIX_FMT_RGB24;

        int frameCount = 0;
        int savedFrames = 0;

        std::cout << "Processing video..." << std::endl;

        while (av_read_frame(fmtCtx, packet) >= 0) {
            if (packet->stream_index == videoStreamIdx) {
                int ret = avcodec_send_packet(codecCtx, packet);
                if (ret < 0) {
                    av_packet_unref(packet);
                    continue;
                }
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codecCtx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    }
                    else if (ret < 0) {
                        std::cerr << "Error during decoding" << std::endl;
                        break;
                    }

                    frameCount++;

                    int h = sws_scale(swsCtx,
                        frame->data, frame->linesize,
                        0, codecCtx->height,
                        rgbFrame->data, rgbFrame->linesize);

                    if (h <= 0) {
                        std::cerr << "sws_scale failed for frame " << frameCount << std::endl;
                    }
                    else {
                        int64_t pts = (frame->pts != AV_NOPTS_VALUE) ? frame->pts : frame->best_effort_timestamp;
                        if (frameCount % frameInterval == 0) {
                            processFrame(rgbFrame, pts);
                            savedFrames++;

                            if (savedFrames % 10 == 0) {
                                std::cout << "Processed " << savedFrames << " frames (total decoded: " << frameCount << ")" << std::endl;
                            }
                        }
                    }

                    av_frame_unref(frame);
                }
            }
            av_packet_unref(packet);
        }

        metadata.totalFrames = savedFrames;
        totalProcessedFrames = frameCount;

        std::cout << "Total frames decoded: " << frameCount << std::endl;
        std::cout << "Frames saved: " << savedFrames << std::endl << std::endl;

        bool result = saveToFile();

        av_free(buffer);
        av_frame_free(&rgbFrame);
        av_frame_free(&frame);
        av_packet_free(&packet);

        return result;
    }

private:
    // Aggressive RLE + dictionary compression
    std::vector<uint8_t> compressKeyframe(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> compressed;
        compressed.reserve(data.size() / 3);

        size_t i = 0;
        while (i < data.size()) {
            // Look for runs
            size_t runLen = 1;
            while (i + runLen < data.size() &&
                data[i + runLen] == data[i] &&
                runLen < 255) {
                runLen++;
            }

            if (runLen >= 4) {
                // RLE: 0xFF + length + value
                compressed.push_back(0xFF);
                compressed.push_back(static_cast<uint8_t>(runLen));
                compressed.push_back(data[i]);
                i += runLen;
            }
            else {
                // Literal run: look for non-repeating sequences
                size_t literalStart = i;
                size_t literalLen = 0;

                while (i < data.size() && literalLen < 127) {
                    // Check if next bytes repeat
                    size_t nextRun = 1;
                    if (i + 1 < data.size()) {
                        while (i + nextRun < data.size() &&
                            data[i + nextRun] == data[i] &&
                            nextRun < 4) {
                            nextRun++;
                        }
                    }

                    if (nextRun >= 4) break; // Start RLE

                    literalLen++;
                    i++;
                }

                // Write literal: length + data
                compressed.push_back(static_cast<uint8_t>(literalLen));
                for (size_t j = 0; j < literalLen; j++) {
                    compressed.push_back(data[literalStart + j]);
                }
            }
        }

        return compressed;
    }

    // Compress delta with variable-length encoding
    std::vector<uint8_t> compressDelta(const std::vector<uint8_t>& current,
        const std::vector<uint8_t>& previous) {
        std::vector<uint8_t> compressed;
        compressed.reserve(current.size() / 10);

        size_t i = 0;
        size_t frameSize = current.size();

        while (i < frameSize) {
            // Find next changed pixel
            while (i < frameSize &&
                std::abs(static_cast<int>(current[i]) - static_cast<int>(previous[i])) < changeThreshold) {
                i++;
            }

            if (i >= frameSize) break;

            // Found a change - count consecutive changes
            size_t changeStart = i;
            std::vector<int8_t> deltas;

            while (i < frameSize && deltas.size() < 255) {
                int diff = static_cast<int>(current[i]) - static_cast<int>(previous[i]);
                if (std::abs(diff) >= changeThreshold) {
                    // Clamp to int8_t
                    int8_t delta = (diff > 127) ? 127 : (diff < -128) ? -128 : static_cast<int8_t>(diff);
                    deltas.push_back(delta);
                    i++;
                }
                else {
                    // Gap in changes - only store if we have changes to write
                    if (deltas.size() >= 3) {
                        break;
                    }
                    // Small gap - include it
                    if (i - changeStart < 5) {
                        deltas.push_back(0);
                        i++;
                    }
                    else {
                        break;
                    }
                }
            }

            if (deltas.empty()) continue;

            // Write: start_index (varint) + length + deltas
            writeVarint(compressed, changeStart);
            compressed.push_back(static_cast<uint8_t>(deltas.size()));
            for (int8_t delta : deltas) {
                compressed.push_back(static_cast<uint8_t>(delta));
            }
        }

        return compressed;
    }

    // Variable-length integer encoding (saves space on small indices)
    void writeVarint(std::vector<uint8_t>& out, size_t value) {
        while (value >= 0x80) {
            out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        out.push_back(static_cast<uint8_t>(value));
    }

    void processFrame(AVFrame* frame, int64_t pts) {
        size_t frameSize = static_cast<size_t>(metadata.width) * static_cast<size_t>(metadata.height) * 3;
        std::vector<uint8_t> currentFrame(frameSize);

        for (int i = 0; i < metadata.height; i++) {
            memcpy(currentFrame.data() + static_cast<size_t>(i) * metadata.width * 3,
                frame->data[0] + static_cast<size_t>(i) * frame->linesize[0],
                static_cast<size_t>(metadata.width) * 3);
        }

        CompressedFrame cf;
        cf.timestamp = pts;

        bool shouldBeKeyframe = (savedFrameCount % keyframeInterval == 0);

        if (lastFrame.empty() || shouldBeKeyframe) {
            cf.frameType = 1; // Keyframe
            cf.data = compressKeyframe(currentFrame);
        }
        else {
            cf.frameType = 0; // Delta
            cf.data = compressDelta(currentFrame, lastFrame);
        }

        compressedFrames.push_back(std::move(cf));
        lastFrame = std::move(currentFrame);
        savedFrameCount++;
    }

    double MP4FileSize(const std::string& filePath) {
        try {
            auto sizeBytes = std::filesystem::file_size(std::filesystem::path(filePath));
            return static_cast<double>(sizeBytes) / (1024.0 * 1024.0);
        }
        catch (const std::exception& e) {
            std::cerr << "Could not get file size: " << e.what() << std::endl;
            return -1.0;
        }
    }

    bool saveToFile() {
        std::cout << "\n=== COMPRESSION DIAGNOSTICS ===" << std::endl;

        size_t keyframeCount = 0;
        size_t totalKeyframeBytes = 0;
        size_t totalDeltaBytes = 0;

        for (const auto& cf : compressedFrames) {
            if (cf.frameType == 1) {
                keyframeCount++;
                totalKeyframeBytes += cf.data.size();
            }
            else {
                totalDeltaBytes += cf.data.size();
            }
        }

        std::cout << "Total frames: " << compressedFrames.size() << std::endl;
        std::cout << "Key frames: " << keyframeCount << " ("
            << totalKeyframeBytes / (1024.0 * 1024.0) << " MB)" << std::endl;
        std::cout << "Delta frames: " << (compressedFrames.size() - keyframeCount)
            << " (" << totalDeltaBytes / (1024.0 * 1024.0) << " MB)" << std::endl;

        std::ofstream ofs(outputFile, std::ios::binary);
        if (!ofs) {
            std::cerr << "Could not open output file: " << outputFile << std::endl;
            return false;
        }

        // Magic header
        const char* magic = "IGEDLT2";  // Version 2
        ofs.write(magic, 8);

        // Metadata
        ofs.write(reinterpret_cast<const char*>(&metadata), sizeof(VideoMetadata));

        // Frame count
        uint32_t frameCount = static_cast<uint32_t>(compressedFrames.size());
        ofs.write(reinterpret_cast<const char*>(&frameCount), sizeof(uint32_t));

        // Frames
        for (const auto& cf : compressedFrames) {
            ofs.write(reinterpret_cast<const char*>(&cf.timestamp), sizeof(int64_t));
            ofs.write(reinterpret_cast<const char*>(&cf.frameType), sizeof(uint8_t));

            uint32_t dataSize = static_cast<uint32_t>(cf.data.size());
            ofs.write(reinterpret_cast<const char*>(&dataSize), sizeof(uint32_t));
            ofs.write(reinterpret_cast<const char*>(cf.data.data()), cf.data.size());
        }

        ofs.close();

        std::cout << "\nCompressed video saved to: " << outputFile << std::endl;

        uint64_t originalSize = 1ULL * metadata.width * metadata.height * 3 *
            static_cast<uint64_t>(totalProcessedFrames);

        std::ifstream in(outputFile, std::ios::ate | std::ios::binary);
        size_t compressedSize = 0;
        if (in) {
            compressedSize = static_cast<size_t>(in.tellg());
            in.close();
        }

        double inputFileMB = MP4FileSize(inputFile);
        double ratio = (compressedSize > 0) ?
            (static_cast<double>(originalSize) / static_cast<double>(compressedSize)) : 0.0;
        double compressionVsMP4 = (inputFileMB > 0 && compressedSize > 0) ?
            (inputFileMB * 1024 * 1024) / static_cast<double>(compressedSize) : 0.0;

        std::cout << "\n=== COMPRESSION RESULTS ===" << std::endl;
        std::cout << "Input MP4 file size: " << inputFileMB << " MB" << std::endl;
        std::cout << "Output .ige file size: " << compressedSize / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Compression vs MP4: " << compressionVsMP4 << ":1 ";

        if (compressionVsMP4 < 1.0) {
            std::cout << "(WARNING: Output is LARGER than original MP4!)" << std::endl;
            std::cout << "\nSuggestions to improve compression:" << std::endl;
            std::cout << "  - Increase change threshold (current: " << changeThreshold << ")" << std::endl;
            std::cout << "  - Increase frame interval (current: " << frameInterval << ")" << std::endl;
            std::cout << "  - Reduce keyframe frequency (current: every " << keyframeInterval << " frames)" << std::endl;
        }
        else {
            std::cout << "(Successfully compressed!)" << std::endl;
        }

        std::cout << "\nVideo resolution: " << metadata.width << "x" << metadata.height << "px" << std::endl;
        std::cout << "Raw sampled size: " << originalSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Sampled frames saved: " << metadata.totalFrames << std::endl;
        std::cout << "Compression ratio (vs raw): " << ratio << ":1" << std::endl;
        std::cout << "Change threshold: " << changeThreshold << std::endl;
        std::cout << "Keyframe interval: every " << keyframeInterval << " frames" << std::endl;

        return true;
    }

    void cleanup() {
        if (swsCtx) sws_freeContext(swsCtx);
        if (codecCtx) avcodec_free_context(&codecCtx);
        if (fmtCtx) avformat_close_input(&fmtCtx);
    }
};

int main(int argc, char* argv[]) {
    std::cout << "Start of program" << std::endl;

    if (argc < 3) {
        std::cout << "Video Delta Compressor" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input.mp4> <output.ige> [frame_interval] [keyframe_interval] [change_threshold]" << std::endl;
        std::cout << "  frame_interval: Sample every Nth frame (default: 10)" << std::endl;
        std::cout << "  keyframe_interval: Keyframe every N frames (default: 30)" << std::endl;
        std::cout << "  change_threshold: Ignore changes < N (default: 15, higher=better compression)" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " video.mp4 compressed.ige 10 30 15" << std::endl;
        std::cout << "  " << argv[0] << " video.mp4 compressed.ige 15 25 20  # More aggressive" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int frameInterval = (argc >= 4) ? std::atoi(argv[3]) : 10;
    int keyframeInterval = (argc >= 5) ? std::atoi(argv[4]) : 30;
    int changeThreshold = (argc >= 6) ? std::atoi(argv[5]) : 15;

    std::cout << "Video Delta Compressor" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input: " << inputFile << std::endl;
    std::cout << "Output: " << outputFile << std::endl;
    std::cout << "Frame interval: " << frameInterval << std::endl;
    std::cout << "Keyframe interval: " << keyframeInterval << std::endl;
    std::cout << "Change threshold: " << changeThreshold << std::endl << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    VideoCompressor compressor(inputFile, outputFile, frameInterval, keyframeInterval, changeThreshold);

    if (!compressor.initialize()) {
        std::cerr << "Failed to initialize compressor" << std::endl;
        return 1;
    }

    if (!compressor.compress()) {
        std::cerr << "Compression failed" << std::endl;
        return 1;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = endTime - startTime;
    long long micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    if (micros < 1000)
        std::cout << "\nTime taken: " << micros << " µs" << std::endl;
    else if (micros < 1'000'000)
        std::cout << "\nTime taken: " << micros / 1000.0 << " ms" << std::endl;
    else
        std::cout << "\nTime taken: " << micros / 1'000'000.0 << " s" << std::endl;

    std::cout << "\nCompression completed successfully!" << std::endl;

    return 0;
}