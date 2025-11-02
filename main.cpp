#include <iostream>
#include <fstream>  
#include <vector>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <filesystem> 
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

struct PixelChange {
    int32_t index;     // Pixel index in the frame
    int16_t delta;     // Change value
};

struct CompressedFrame {
    int64_t timestamp;
    std::vector<PixelChange> changes; 
};

struct VideoMetadata {
    int width;
    int height;
    int frameInterval; // Save every Nth frame
    int64_t duration;
    int totalFrames;
};

class VideoCompressor {
private:
    std::string inputFile;
    std::string outputFile;
    int frameInterval;
    int totalProcessedFrames = 0;

    AVFormatContext* fmtCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    SwsContext* swsCtx = nullptr;
    int videoStreamIdx = -1;

    VideoMetadata metadata{};
    std::vector<CompressedFrame> compressedFrames;
    std::vector<uint8_t> lastFrame;
    const int CHANGE_THRESHOLD = 5; // Only store changes >= 5

public:
    /**
     * Constructor: Initializes the video compressor with input/output files
     * @param input - Path to input MP4 file
     * @param output - Path to output compressed file (.ige)
     * @param interval - Process every Nth frame (default: 10)
     */

    VideoCompressor(const std::string& input, const std::string& output, int interval = 10)
        : inputFile(input), outputFile(output), frameInterval(interval) {
    }

    /**
     * Destructor: Ensures all FFmpeg resources are properly released
     */

    ~VideoCompressor() {
        cleanup();
    }

    /**
     * Initializes FFmpeg components and opens the input video file
     * Sets up codec context, format context, and scaling context for RGB conversion
     * @return true if initialization successful, false otherwise
     */

    bool initialize() {
        // Open input file
        if (avformat_open_input(&fmtCtx, inputFile.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Could not open input file: " << inputFile << std::endl;
            return false;
        }

        // Retrieve stream information
        if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
            std::cerr << "Could not find stream information" << std::endl;
            return false;
        }

        // Find video stream
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

        // Get codec parameters
        AVCodecParameters* codecParams = fmtCtx->streams[videoStreamIdx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);

        if (!codec) {
            std::cerr << "Unsupported codec" << std::endl;
            return false;
        }

        // Create codec context
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

        // Store metadata
        metadata.width = codecCtx->width;
        metadata.height = codecCtx->height;
        metadata.frameInterval = frameInterval;
        metadata.duration = fmtCtx->duration;
        metadata.totalFrames = 0;

        // Initialize SWS context for RGB conversion
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

        // Allocate buffer for RGB frame
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

        while (av_read_frame(fmtCtx, packet) >= 0) {
            if (packet->stream_index == videoStreamIdx) {
                int ret = avcodec_send_packet(codecCtx, packet);
                if (ret < 0) {
                    // non-fatal: continue to next packet
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

                    frameCount++; // count every decoded frame

                    // Convert decoded frame to RGB into rgbFrame
                    int h = sws_scale(swsCtx,
                        frame->data, frame->linesize,
                        0, codecCtx->height,
                        rgbFrame->data, rgbFrame->linesize);

                    if (h <= 0) {
                        std::cerr << "sws_scale failed for frame " << frameCount << std::endl;
                    } else {
                        // Use pts if available; fallback to best_effort_timestamp
                        int64_t pts = (frame->pts != AV_NOPTS_VALUE) ? frame->pts : frame->best_effort_timestamp;
                        if (frameCount % frameInterval == 0) {
                            // process rgb frame
                            processFrame(rgbFrame, pts);
                            savedFrames++;
                        }
                    }

                    av_frame_unref(frame);
                }
            }
            av_packet_unref(packet);
        }

        // Save metadata
        metadata.totalFrames = savedFrames;          // frames actually saved
        totalProcessedFrames = frameCount;          // total frames decoded

        std::cout << "Total frames processed: " << frameCount << std::endl;
        std::cout << "Frames saved: " << savedFrames << std::endl << std::endl;

        // Save to file
        bool result = saveToFile();

        // Cleanup
        av_free(buffer);
        av_frame_free(&rgbFrame);
        av_frame_free(&frame);
        av_packet_free(&packet);

        return result;
    }

private:
    void processFrame(AVFrame* frame, int64_t pts) {
       
        size_t frameSize = static_cast<size_t>(metadata.width) * static_cast<size_t>(metadata.height) * 3; // RGB24
        std::vector<uint8_t> currentFrame(frameSize);

        // Copy frame data row by row 
        for (int i = 0; i < metadata.height; i++) {
            memcpy(currentFrame.data() + static_cast<size_t>(i) * metadata.width * 3,
                frame->data[0] + static_cast<size_t>(i) * frame->linesize[0],
                static_cast<size_t>(metadata.width) * 3);
        }

        CompressedFrame cf;
        cf.timestamp = pts;

        if (lastFrame.empty()) {
            // First frame - store all bytes as changes (store as delta from zero)
            for (size_t i = 0; i < frameSize; i++) {
                PixelChange pc;
                pc.index = static_cast<int32_t>(i);
                pc.delta = static_cast<int16_t>(currentFrame[i]); // delta from 0
                cf.changes.push_back(pc);
            }
        }
        else {
            // Store only significant changes (|delta| >= CHANGE_THRESHOLD)
            for (size_t i = 0; i < frameSize; i++) {
                int16_t diff = static_cast<int16_t>(currentFrame[i]) - static_cast<int16_t>(lastFrame[i]);

                if (std::abs(diff) >= CHANGE_THRESHOLD) {
                    PixelChange pc;
                    pc.index = static_cast<int32_t>(i);
                    pc.delta = diff;
                    cf.changes.push_back(pc);
                }
            }
        }

        compressedFrames.push_back(std::move(cf));
        lastFrame = std::move(currentFrame);
    }

    double MP4FileSize(const std::string& filePath) {
        try {
            auto sizeBytes = std::filesystem::file_size(std::filesystem::path(filePath));
            double sizeMB = static_cast<double>(sizeBytes) / (1024.0 * 1024.0);
            return sizeMB;
        }
        catch (const std::exception& e) {
            std::cerr << "Could not open file " << filePath << ": " << e.what() << std::endl;
            return -1.0;
        }
    }

    bool saveToFile() {
        std::ofstream ofs(outputFile, std::ios::binary);
        if (!ofs) {
            std::cerr << "Could not open output file: " << outputFile << std::endl;
            return false;
        }

        // Write magic header
        const char* magic = "IGEDELTA";
        ofs.write(magic, 8);

        // Write metadata
        ofs.write(reinterpret_cast<const char*>(&metadata), sizeof(VideoMetadata));

        // Write compressed frames
        size_t totalChanges = 0;
        for (const auto& cf : compressedFrames) {
            // Write timestamp
            ofs.write(reinterpret_cast<const char*>(&cf.timestamp), sizeof(int64_t));

            // Write number of changes
            size_t numChanges = cf.changes.size();
            ofs.write(reinterpret_cast<const char*>(&numChanges), sizeof(size_t));

            // Write all changes
            for (const auto& change : cf.changes) {
                ofs.write(reinterpret_cast<const char*>(&change.index), sizeof(int32_t));
                ofs.write(reinterpret_cast<const char*>(&change.delta), sizeof(int16_t));
            }

            totalChanges += numChanges;
        }

        ofs.close();

        std::cout << "Compressed video saved to: " << outputFile << std::endl;

        // Calculate compression stats
        uint64_t originalSize = 1ULL * metadata.width * metadata.height * 3 * static_cast<uint64_t>(totalProcessedFrames);
        std::ifstream in(outputFile, std::ios::ate | std::ios::binary);
        size_t compressedSize = 0;
        if (in) {
            compressedSize = static_cast<size_t>(in.tellg());
            in.close();
        } else {
            std::cerr << "Could not read compressed file size for " << outputFile << std::endl;
        }

        double inputFileMB = MP4FileSize(inputFile);

        double ratio = (compressedSize > 0) ? (static_cast<double>(originalSize) / static_cast<double>(compressedSize)) : 0.0;
        float avgChangesPerFrame = (metadata.totalFrames > 0) ? (static_cast<float>(totalChanges) / static_cast<float>(metadata.totalFrames)) : 0.0f;
        float changePercentage = 0.0f;
        if (metadata.width > 0 && metadata.height > 0) {
            float totalBytesPerFrame = static_cast<float>(metadata.width) * metadata.height * 3.0f;
            changePercentage = (totalBytesPerFrame > 0.0f) ? ((avgChangesPerFrame / totalBytesPerFrame) * 100.0f) : 0.0f;
        }

        std::cout << "Input file size: " << (inputFileMB >= 0.0 ? std::to_string(inputFileMB) + " MB" : "unknown") << std::endl;
        std::cout << "Original size : " << originalSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Video resolution: " << metadata.width << "px, " << metadata.height << "px " << std::endl;
        std::cout << "Compressed size: " << compressedSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Compression ratio: " << ratio << ":1" << std::endl;
        std::cout << "Avg changes per frame: " << avgChangesPerFrame << " (" << changePercentage << "% of bytes)" << std::endl;

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
        std::cout << "Usage:  <input.mp4> <output.ige> [frame_interval]" << std::endl;
        std::cout << "Frame_interval: Save every Nth frame (default: 10)" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int frameInterval = (argc >= 4) ? std::atoi(argv[3]) : 10;

    std::cout << "Video Delta Compressor" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Input: " << inputFile << std::endl;
    std::cout << "Output: " << outputFile << std::endl;
    std::cout << "Frame interval: " << frameInterval << std::endl << std::endl;


    std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
    VideoCompressor compressor(inputFile, outputFile, frameInterval);

    if (!compressor.initialize()) {
        std::cerr << "Failed to initialize compressor" << std::endl;
        return 1;
    }

    if (!compressor.compress()) {
        std::cerr << "Compression failed" << std::endl;
        return 1;
    }


    std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration elapsed = endTime - startTime;

    // Converts time to microseconds
    long long micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    // Determines which format the time should be displayed
    if (micros < 1000)
        std::cout << "Time taken: " << micros << " µs\n";
    else if (micros < 1'000'000)
        std::cout << "Time taken: " << micros / 1000.0 << " ms\n";
    else
        std::cout << "Time taken: " << micros / 1'000'000.0 << " s\n";

    std::cout << "\nCompression completed successfully!\n";

    return 0;
}
