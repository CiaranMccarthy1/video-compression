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
#include <memory>
#include <functional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

// --- RAII Wrappers for FFmpeg ---

struct AVFormatContextDeleter {
    void operator()(AVFormatContext* ctx) const {
        if (ctx) avformat_close_input(&ctx);
    }
};
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;

struct AVCodecContextDeleter {
    void operator()(AVCodecContext* ctx) const {
        if (ctx) avcodec_free_context(&ctx);
    }
};
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;

struct AVFrameDeleter {
    void operator()(AVFrame* frame) const {
        if (frame) av_frame_free(&frame);
    }
};
using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

struct AVPacketDeleter {
    void operator()(AVPacket* pkt) const {
        if (pkt) av_packet_free(&pkt);
    }
};
using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;

struct SwsContextDeleter {
    void operator()(SwsContext* ctx) const {
        if (ctx) sws_freeContext(ctx);
    }
};
using SwsContextPtr = std::unique_ptr<SwsContext, SwsContextDeleter>;

// --- Data Structures ---

struct VideoMetadata {
    int width;
    int height;
    int frameInterval;
    int64_t duration;
    int totalFrames;
    int keyframeInterval;
    int changeThreshold;
};

// --- Compressor Class ---

class VideoCompressor {
private:
    std::string inputFile;
    std::string outputFile;
    int frameInterval;
    int keyframeInterval;
    int changeThreshold;

    int totalProcessedFrames = 0;
    int savedFrameCount = 0;
    
    // Statistics
    size_t keyframeCount = 0;
    size_t totalKeyframeBytes = 0;
    size_t totalDeltaBytes = 0;

    // Resources
    AVFormatContextPtr fmtCtx;
    AVCodecContextPtr codecCtx;
    SwsContextPtr swsCtx;
    int videoStreamIdx = -1;

    VideoMetadata metadata{};
    std::vector<uint8_t> lastFrame;
    std::ofstream ofs;
    std::streampos headerPos;

public:
    VideoCompressor(const std::string& input, const std::string& output,
        int interval = 10, int kfInterval = 30, int threshold = 15)
        : inputFile(input), outputFile(output), frameInterval(interval),
        keyframeInterval(kfInterval), changeThreshold(threshold) {
    }

    ~VideoCompressor() {
    }

    bool initialize() {
        // 1. Open Input
        AVFormatContext* rawFmtCtx = nullptr;
        if (avformat_open_input(&rawFmtCtx, inputFile.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Could not open input file: " << inputFile << std::endl;
            return false;
        }
        fmtCtx.reset(rawFmtCtx);

        if (avformat_find_stream_info(fmtCtx.get(), nullptr) < 0) {
            std::cerr << "Could not find stream information" << std::endl;
            return false;
        }

        // 2. Find Video Stream
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

        // 3. Setup Codec
        AVCodecParameters* codecParams = fmtCtx->streams[videoStreamIdx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);

        if (!codec) {
            std::cerr << "Unsupported codec" << std::endl;
            return false;
        }

        codecCtx.reset(avcodec_alloc_context3(codec));
        if (!codecCtx) {
            std::cerr << "Could not allocate codec context" << std::endl;
            return false;
        }

        if (avcodec_parameters_to_context(codecCtx.get(), codecParams) < 0) {
            std::cerr << "Could not copy codec params to context" << std::endl;
            return false;
        }

        if (avcodec_open2(codecCtx.get(), codec, nullptr) < 0) {
            std::cerr << "Could not open codec" << std::endl;
            return false;
        }

        // 4. Setup Metadata
        metadata.width = codecCtx->width;
        metadata.height = codecCtx->height;
        metadata.frameInterval = frameInterval;
        metadata.keyframeInterval = keyframeInterval;
        metadata.changeThreshold = changeThreshold;
        metadata.duration = fmtCtx->duration;
        metadata.totalFrames = 0; 

        // 5. Setup SwsContext (Convert to YUV420P)
        swsCtx.reset(sws_getContext(
            codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
            codecCtx->width, codecCtx->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        ));

        if (!swsCtx) {
            std::cerr << "Could not initialize SWS context" << std::endl;
            return false;
        }

        // 6. Init Output File
        ofs.open(outputFile, std::ios::binary);
        if (!ofs) {
            std::cerr << "Could not open output file: " << outputFile << std::endl;
            return false;
        }

        writeHeader(); // Write placeholder

        return true;
    }

    bool compress() {
        AVPacketPtr packet(av_packet_alloc());
        AVFramePtr frame(av_frame_alloc());
        AVFramePtr scaledFrame(av_frame_alloc()); // Was rgbFrame, now generic

        if (!packet || !frame || !scaledFrame) {
            std::cerr << "Could not allocate packet/frame" << std::endl;
            return false;
        }

        // Allocate buffer for YUV420P frame
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, codecCtx->width, codecCtx->height, 1);
        
        struct AvFreeDeleter { void operator()(void* p) const { av_free(p); } };
        std::unique_ptr<uint8_t, AvFreeDeleter> buffer((uint8_t*)av_malloc((size_t)numBytes));
        
        if (!buffer) {
            std::cerr << "Could not allocate image buffer" << std::endl;
            return false;
        }
        
        av_image_fill_arrays(scaledFrame->data, scaledFrame->linesize, buffer.get(), AV_PIX_FMT_YUV420P, codecCtx->width, codecCtx->height, 1);
        scaledFrame->width = codecCtx->width;
        scaledFrame->height = codecCtx->height;
        scaledFrame->format = AV_PIX_FMT_YUV420P;

        int frameCount = 0;
        
        std::cout << "Processing video..." << std::endl;

        while (av_read_frame(fmtCtx.get(), packet.get()) >= 0) {
            if (packet->stream_index == videoStreamIdx) {
                int ret = avcodec_send_packet(codecCtx.get(), packet.get());
                if (ret < 0) {
                    av_packet_unref(packet.get());
                    continue; 
                }
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codecCtx.get(), frame.get());
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    }
                    else if (ret < 0) {
                        std::cerr << "Error during decoding" << std::endl;
                        break;
                    }

                    frameCount++;

                    // Scale to YUV420P
                    int h = sws_scale(swsCtx.get(),
                        frame->data, frame->linesize,
                        0, codecCtx->height,
                        scaledFrame->data, scaledFrame->linesize);

                    if (h <= 0) {
                        std::cerr << "sws_scale failed for frame " << frameCount << std::endl;
                    }
                    else {
                        int64_t pts = (frame->pts != AV_NOPTS_VALUE) ? frame->pts : frame->best_effort_timestamp;
                        
                        if (frameCount % frameInterval == 0) {
                            processAndWriteFrame(scaledFrame.get(), pts);
                            
                            if (savedFrameCount % 10 == 0) {
                                std::cout << "Processed " << savedFrameCount << " frames (total decoded: " << frameCount << ")" << std::endl;
                            }
                        }
                    }
                    av_frame_unref(frame.get());
                }
            }
            av_packet_unref(packet.get());
        }

        metadata.totalFrames = savedFrameCount;
        totalProcessedFrames = frameCount; 

        finalizeFile();
        printStats();

        return true;
    }

private:
    void writeHeader() {
        const char* magic = "IGEDLT2";
        ofs.write(magic, 8);

        headerPos = ofs.tellp();
        
        // Metadata placeholder
        ofs.write(reinterpret_cast<const char*>(&metadata), sizeof(VideoMetadata));

        // Frame count placeholder
        uint32_t count = 0;
        ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }

    void finalizeFile() {
        if (!ofs) return;

        ofs.seekp(headerPos);
        ofs.write(reinterpret_cast<const char*>(&metadata), sizeof(VideoMetadata));
        
        uint32_t count = static_cast<uint32_t>(savedFrameCount);
        ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        
        ofs.close();
    }

    void processAndWriteFrame(AVFrame* frame, int64_t pts) {
        int w = metadata.width;
        int h = metadata.height;
        int uvw = w / 2;
        int uvh = h / 2;
        
        size_t ySize = w * h;
        size_t uvSize = uvw * uvh;
        size_t totalSize = ySize + 2 * uvSize;

        std::vector<uint8_t> currentFrame(totalSize);
        uint8_t* dst = currentFrame.data();

        // Copy Y plane
        // frame->data[0] is Y, linesize[0] is stride
        for (int i = 0; i < h; i++) {
            memcpy(dst + i * w, frame->data[0] + i * frame->linesize[0], w);
        }
        dst += ySize;

        // Copy U plane
        for (int i = 0; i < uvh; i++) {
            memcpy(dst + i * uvw, frame->data[1] + i * frame->linesize[1], uvw);
        }
        dst += uvSize;

        // Copy V plane
        for (int i = 0; i < uvh; i++) {
            memcpy(dst + i * uvw, frame->data[2] + i * frame->linesize[2], uvw);
        }

        // Now currentFrame contains tightly packed YUV data
        uint8_t frameType = 0;
        std::vector<uint8_t> compressedData;

        bool shouldBeKeyframe = (savedFrameCount % keyframeInterval == 0);

        if (lastFrame.empty() || shouldBeKeyframe) {
            frameType = 1; // Keyframe
            compressedData = compressKeyframe(currentFrame);
            
            keyframeCount++;
            totalKeyframeBytes += compressedData.size();
        }
        else {
            frameType = 0; // Delta
            compressedData = compressDelta(currentFrame, lastFrame);
            
            totalDeltaBytes += compressedData.size();
        }

        writeFrameToDisk(pts, frameType, compressedData);

        lastFrame = std::move(currentFrame);
        savedFrameCount++;
    }

    void writeFrameToDisk(int64_t pts, uint8_t frameType, const std::vector<uint8_t>& data) {
        ofs.write(reinterpret_cast<const char*>(&pts), sizeof(int64_t));
        ofs.write(reinterpret_cast<const char*>(&frameType), sizeof(uint8_t));

        uint32_t dataSize = static_cast<uint32_t>(data.size());
        ofs.write(reinterpret_cast<const char*>(&dataSize), sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    std::vector<uint8_t> compressKeyframe(const std::vector<uint8_t>& data) {
        // Reuse existing RLE logic - it works on any byte byte sequence
        std::vector<uint8_t> compressed;
        compressed.reserve(data.size() / 2); // Better estimate for YUV

        size_t i = 0;
        while (i < data.size()) {
            size_t runLen = 1;
            while (i + runLen < data.size() && data[i + runLen] == data[i] && runLen < 255) {
                runLen++;
            }

            if (runLen >= 4) {
                compressed.push_back(0xFF);
                compressed.push_back(static_cast<uint8_t>(runLen));
                compressed.push_back(data[i]);
                i += runLen;
            }
            else {
                size_t literalStart = i;
                size_t literalLen = 0;
                while (i < data.size() && literalLen < 127) {
                    size_t nextRun = 1;
                    if (i + 1 < data.size()) {
                        while (i + nextRun < data.size() && data[i + nextRun] == data[i] && nextRun < 4) nextRun++;
                    }
                    if (nextRun >= 4) break; 
                    literalLen++;
                    i++;
                }
                compressed.push_back(static_cast<uint8_t>(literalLen));
                for (size_t j = 0; j < literalLen; j++) compressed.push_back(data[literalStart + j]);
            }
        }
        return compressed;
    }

    std::vector<uint8_t> compressDelta(const std::vector<uint8_t>& current, const std::vector<uint8_t>& previous) {
        std::vector<uint8_t> compressed;
        compressed.reserve(current.size() / 10);

        size_t i = 0;
        size_t frameSize = current.size();

        while (i < frameSize) {
            while (i < frameSize &&
                std::abs(static_cast<int>(current[i]) - static_cast<int>(previous[i])) < changeThreshold) {
                i++;
            }

            if (i >= frameSize) break;

            size_t changeStart = i;
            std::vector<int8_t> deltas;

            while (i < frameSize && deltas.size() < 255) {
                int diff = static_cast<int>(current[i]) - static_cast<int>(previous[i]);
                if (std::abs(diff) >= changeThreshold) {
                    int8_t delta = (diff > 127) ? 127 : (diff < -128) ? -128 : static_cast<int8_t>(diff);
                    deltas.push_back(delta);
                    i++;
                }
                else {
                    if (deltas.size() >= 3) break;
                    if (i - changeStart < 5) {
                        deltas.push_back(0);
                        i++;
                    }
                    else break;
                }
            }

            if (deltas.empty()) continue;

            writeVarint(compressed, changeStart);
            compressed.push_back(static_cast<uint8_t>(deltas.size()));
            for (int8_t delta : deltas) compressed.push_back(static_cast<uint8_t>(delta));
        }
        return compressed;
    }

    void writeVarint(std::vector<uint8_t>& out, size_t value) {
        while (value >= 0x80) {
            out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        out.push_back(static_cast<uint8_t>(value));
    }

    double MP4FileSize(const std::string& filePath) {
        try {
            auto sizeBytes = std::filesystem::file_size(std::filesystem::path(filePath));
            return static_cast<double>(sizeBytes) / (1024.0 * 1024.0);
        } catch (...) { return -1.0; }
    }

    void printStats() {
        std::cout << "\n=== COMPRESSION DIAGNOSTICS ===" << std::endl;
        std::cout << "Total frames: " << savedFrameCount << std::endl;
        std::cout << "Key frames: " << keyframeCount << " ("
            << totalKeyframeBytes / (1024.0 * 1024.0) << " MB)" << std::endl;
        std::cout << "Delta frames: " << (savedFrameCount - keyframeCount)
            << " (" << totalDeltaBytes / (1024.0 * 1024.0) << " MB)" << std::endl;

        std::cout << "\nCompressed video saved to: " << outputFile << std::endl;

        // Raw size for YUV420P is W * H * 1.5 per frame
        // Safely calculate: (W * H * 3) / 2
        uint64_t pixels = static_cast<uint64_t>(metadata.width) * metadata.height;
        uint64_t originalSize = (pixels * 3 * totalProcessedFrames) / 2;
        
        double compressedSizeMB = MP4FileSize(outputFile);
        double inputFileMB = MP4FileSize(inputFile);
        size_t compressedSizeBytes = (size_t)(compressedSizeMB * 1024.0 * 1024.0);

        double ratio = (compressedSizeBytes > 0) ?
            (static_cast<double>(originalSize) / static_cast<double>(compressedSizeBytes)) : 0.0;
        double compressionVsMP4 = (inputFileMB > 0 && compressedSizeBytes > 0) ?
            (inputFileMB * 1024 * 1024) / static_cast<double>(compressedSizeBytes) : 0.0;

        std::cout << "\n=== COMPRESSION RESULTS ===" << std::endl;
        std::cout << "Input MP4 file size: " << inputFileMB << " MB" << std::endl;
        std::cout << "Output .ige file size: " << compressedSizeMB << " MB" << std::endl;
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
    }
};

int main(int argc, char* argv[]) {
    // Standard setup same as before
    std::cout << "Start of program" << std::endl;

    if (argc < 3) {
        std::cout << "Video Delta Compressor (YUV420P Streaming)" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input.mp4> <output.ige> [frame_interval] [keyframe_interval] [change_threshold]" << std::endl;
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
    
    {
        VideoCompressor compressor(inputFile, outputFile, frameInterval, keyframeInterval, changeThreshold);

        if (!compressor.initialize()) {
            std::cerr << "Failed to initialize compressor" << std::endl;
            return 1;
        }

        if (!compressor.compress()) {
            std::cerr << "Compression failed" << std::endl;
            return 1;
        }
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