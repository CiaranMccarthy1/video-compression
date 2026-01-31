#include "VideoCompressor.h"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <map>
#include <cstring>
#include <chrono>

// -----------------------------------------------------------------------------
// Helper Functions (Internal)
// -----------------------------------------------------------------------------

/** Global state for HW Pixel Format negotiation. Use with care. */
static enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;

/**
 * @brief Callback function used by FFmpeg to negotiate the hardware pixel format.
 * 
 * @param ctx The codec context.
 * @param pix_fmts The list of available pixel formats supplied by the decoder.
 * @return The selected hardware pixel format or AV_PIX_FMT_NONE if not found.
 */
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }
    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

// -----------------------------------------------------------------------------
// Video Compressor Implementation
// -----------------------------------------------------------------------------

VideoCompressor::VideoCompressor(const std::string& input, const std::string& output,
    int interval, int kfInterval, int threshold, int dstW, int dstH, int quantize, bool cuda)
    : inputFile(input), outputFile(output), frameInterval(interval),
    keyframeInterval(kfInterval), changeThreshold(threshold), 
    targetWidth(dstW), targetHeight(dstH), quantization(quantize), useCuda(cuda) {
    if (quantization < 0) quantization = 0;
    if (quantization > 4) quantization = 4;
}

VideoCompressor::~VideoCompressor() {
    // Resources are automatically cleaned up by unique_ptr deleters
}

bool VideoCompressor::initialize() {
    // Reset global HW state to prevent carry-over from previous runs
    hw_pix_fmt = AV_PIX_FMT_NONE;

    // Open input file
    AVFormatContext* rawFmtCtx = nullptr;
    if (avformat_open_input(&rawFmtCtx, inputFile.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file: " << inputFile << std::endl;
        return false;
    }
    fmtCtx.reset(rawFmtCtx);

    // Retrieve stream information
    if (avformat_find_stream_info(fmtCtx.get(), nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        return false;
    }

    // Locate the primary video stream
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

    // Initialize Codec Context
    AVCodecParameters* codecParams = fmtCtx->streams[videoStreamIdx]->codecpar;
    const AVCodec* codec = nullptr; 

    if (useCuda) {
        // In a production environment, we would explicitly iterate to find 'h264_cuvid' or similar.
        // For general purposes, finding the default decoder logic handles negotiation callback.
        codec = avcodec_find_decoder(codecParams->codec_id); 
    } else {
            codec = avcodec_find_decoder(codecParams->codec_id);
    }

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

    // Initialize Hardware Device (if CUDA requested)
    if (useCuda) {
        AVBufferRef* ctx = nullptr;
        int err = av_hwdevice_ctx_create(&ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
        if (err < 0) {
            std::cerr << "Failed to create CUDA device. Error code: " << err << std::endl;
            std::cerr << "Falling back to software decoding." << std::endl;
            useCuda = false;
        } else {
            hwDeviceCtx.reset(ctx);
            codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
            
            // Configure negotiation callback
            hw_pix_fmt = AV_PIX_FMT_CUDA;
            codecCtx->get_format = get_hw_format;
            std::cout << "[Info] CUDA hardware acceleration enabled." << std::endl;
        }
    }

    if (avcodec_open2(codecCtx.get(), codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return false;
    }

    // Store preliminary metadata
    metadata.width = (targetWidth > 0) ? targetWidth : codecCtx->width;
    metadata.height = (targetHeight > 0) ? targetHeight : codecCtx->height;
    metadata.frameInterval = frameInterval;
    metadata.keyframeInterval = keyframeInterval;
    metadata.changeThreshold = changeThreshold;
    metadata.duration = fmtCtx->duration;
    metadata.totalFrames = 0; // Will be updated during finalization

    // Open Output File for Writing
    ofs.open(outputFile, std::ios::binary);
    if (!ofs) {
        std::cerr << "Could not open output file: " << outputFile << std::endl;
        return false;
    }

    writeHeader(); // Write initial header with placeholders

    return true;
}

bool VideoCompressor::compress() {
    AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    AVFramePtr swFrame(av_frame_alloc());     // Intermediate buffer for HW download
    AVFramePtr scaledFrame(av_frame_alloc()); // Final buffer for YUV420P data

    if (!packet || !frame || !swFrame || !scaledFrame) {
        std::cerr << "Could not allocate packet/frame" << std::endl;
        return false;
    }

    // Prepare destination buffer for YUV420P
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, metadata.width, metadata.height, 1);
    struct AvFreeDeleter { void operator()(void* p) const { av_free(p); } };
    std::unique_ptr<uint8_t, AvFreeDeleter> buffer((uint8_t*)av_malloc((size_t)numBytes));
    
    if (!buffer) {
        std::cerr << "Could not allocate image buffer" << std::endl;
        return false;
    }
    
    av_image_fill_arrays(scaledFrame->data, scaledFrame->linesize, buffer.get(), AV_PIX_FMT_YUV420P, metadata.width, metadata.height, 1);
    scaledFrame->width = metadata.width;
    scaledFrame->height = metadata.height;
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
                
                AVFrame* processingFrame = frame.get();

                // Handle Hardware Transfer (GPU -> CPU)
                if (frame->format == hw_pix_fmt) {
                        int err = av_hwframe_transfer_data(swFrame.get(), frame.get(), 0);
                        if (err < 0) {
                            char errbuf[AV_ERROR_MAX_STRING_SIZE];
                            av_strerror(err, errbuf, AV_ERROR_MAX_STRING_SIZE);
                            std::cerr << "Error transferring the data to system memory: " << errbuf << " (" << err << ")" << std::endl;
                            // Signal error so main() can trigger fallback
                            cudaErrorOccurred = true;
                            av_frame_unref(frame.get());
                            return false; 
                        }
                        // Propagate timestamp/properties to SW frame
                        swFrame->pts = frame->pts;
                        swFrame->best_effort_timestamp = frame->best_effort_timestamp;
                        processingFrame = swFrame.get();
                }

                // Dynamically Initialize/Update Scaling Context
                // We check if dimensions or format have changed to handle dynamic resolution changes
                if (!swsCtx || 
                    currentSwsW != processingFrame->width || 
                    currentSwsH != processingFrame->height ||
                    currentSwsFormat != processingFrame->format) {
                        
                    swsCtx.reset(sws_getContext(
                        processingFrame->width, processingFrame->height, (enum AVPixelFormat)processingFrame->format,
                        metadata.width, metadata.height, AV_PIX_FMT_YUV420P,
                        SWS_BILINEAR, nullptr, nullptr, nullptr
                    ));
                    
                    currentSwsW = processingFrame->width;
                    currentSwsH = processingFrame->height;
                    currentSwsFormat = (enum AVPixelFormat)processingFrame->format;
                }

                // Convert frame to YUV420P
                int h = sws_scale(swsCtx.get(),
                    processingFrame->data, processingFrame->linesize,
                    0, processingFrame->height,
                    scaledFrame->data, scaledFrame->linesize);

                if (h <= 0) {
                    std::cerr << "sws_scale failed for frame " << frameCount << std::endl;
                }
                else {
                    int64_t pts = (processingFrame->pts != AV_NOPTS_VALUE) ? processingFrame->pts : processingFrame->best_effort_timestamp;
                    
                    // Sample frames based on interval
                    if (frameCount % frameInterval == 0) {
                        processAndWriteFrame(scaledFrame.get(), pts);
                        
                        if (savedFrameCount % 10 == 0) {
                            std::cout << "Processed " << savedFrameCount << " frames (total decoded: " << frameCount << ")    \r" << std::flush;
                        }
                    }
                }
                av_frame_unref(frame.get());
                av_frame_unref(swFrame.get());
            }
        }
        av_packet_unref(packet.get());
    }

    std::cout << std::endl; // Ensure cursor moves to next line after processing completes

    // Finalize statistics
    metadata.totalFrames = savedFrameCount;
    totalProcessedFrames = frameCount; 

    finalizeFile();
    printStats();

    return true;
}

bool VideoCompressor::hasCudaError() const { return cudaErrorOccurred; }

void VideoCompressor::writeHeader() {
    const char* magic = "IGEDLT2";
    ofs.write(magic, 8);

    headerPos = ofs.tellp();
    
    // Write placeholders for metadata and frame count.
    // We will seek back here to overwrite with real values later.
    ofs.write(reinterpret_cast<const char*>(&metadata), sizeof(VideoMetadata));
    uint32_t count = 0;
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
}

void VideoCompressor::finalizeFile() {
    if (!ofs) return;

    ofs.seekp(headerPos);
    ofs.write(reinterpret_cast<const char*>(&metadata), sizeof(VideoMetadata));
    
    uint32_t count = static_cast<uint32_t>(savedFrameCount);
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    
    ofs.close();
}

void VideoCompressor::processAndWriteFrame(AVFrame* frame, int64_t pts) {
    int w = metadata.width;
    int h = metadata.height;
    int uvw = w / 2;
    int uvh = h / 2;
    
    size_t ySize = w * h;
    size_t uvSize = uvw * uvh;
    size_t totalSize = ySize + 2 * uvSize;

    std::vector<uint8_t> currentFrame(totalSize);
    uint8_t* dst = currentFrame.data();

    // Flatten planar YUV data into a single continuous buffer
    
    // Copy Y plane
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

    // Determine frame type (Keyframe vs Delta)
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

void VideoCompressor::writeFrameToDisk(int64_t pts, uint8_t frameType, const std::vector<uint8_t>& data) {
    ofs.write(reinterpret_cast<const char*>(&pts), sizeof(int64_t));
    ofs.write(reinterpret_cast<const char*>(&frameType), sizeof(uint8_t));

    uint32_t dataSize = static_cast<uint32_t>(data.size());
    ofs.write(reinterpret_cast<const char*>(&dataSize), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
}

std::vector<uint8_t> VideoCompressor::compressKeyframe(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> compressed;
    compressed.reserve(data.size() / 2);

    uint8_t mask = 0xFF << quantization; 

    size_t i = 0;
    while (i < data.size()) {
        size_t runLen = 1;
        uint8_t val = data[i] & mask;

        while (i + runLen < data.size() && (data[i + runLen] & mask) == val && runLen < 255) {
            runLen++;
        }

        if (runLen >= 4) {
            // RLE Sequence: Marker(0xFF) + Length + Value
            compressed.push_back(0xFF);
            compressed.push_back(static_cast<uint8_t>(runLen));
            compressed.push_back(val);
            i += runLen;
        }
        else {
            // Literal Sequence
            size_t literalStart = i;
            size_t literalLen = 0;
            while (i < data.size() && literalLen < 127) {
                size_t nextRun = 1;
                uint8_t currentVal = data[i] & mask;
                
                if (i + 1 < data.size()) {
                    while (i + nextRun < data.size() && (data[i + nextRun] & mask) == currentVal && nextRun < 4) nextRun++;
                }
                if (nextRun >= 4) break; 
                literalLen++;
                i++;
            }
            compressed.push_back(static_cast<uint8_t>(literalLen));
            for (size_t j = 0; j < literalLen; j++) compressed.push_back(data[literalStart + j] & mask);
        }
    }
    return compressed;
}

std::vector<uint8_t> VideoCompressor::compressDelta(const std::vector<uint8_t>& current, const std::vector<uint8_t>& previous) {
    std::vector<uint8_t> compressed;
    compressed.reserve(current.size() / 10);

    uint8_t mask = 0xFF << quantization;

    size_t i = 0;
    size_t frameSize = current.size();

    while (i < frameSize) {
        // Skip pixels with changes below threshold
        while (i < frameSize &&
            std::abs(static_cast<int>(current[i] & mask) - static_cast<int>(previous[i] & mask)) < changeThreshold) {
            i++;
        }

        if (i >= frameSize) break;

        size_t changeStart = i;
        std::vector<int8_t> deltas;

        // Collect consecutive changes
        while (i < frameSize && deltas.size() < 255) {
            int diff = static_cast<int>(current[i] & mask) - static_cast<int>(previous[i] & mask);
            if (std::abs(diff) >= changeThreshold) {
                int8_t delta = (diff > 127) ? 127 : (diff < -128) ? -128 : static_cast<int8_t>(diff);
                deltas.push_back(delta);
                i++;
            }
            else {
                // Allow small gaps to keep runs together
                if (deltas.size() >= 3) break;
                if (i - changeStart < 5) {
                    deltas.push_back(0);
                    i++;
                }
                else break;
            }
        }

        if (deltas.empty()) continue;

        // Write change packet: Position (VarInt) + Length + Delta Values
        writeVarint(compressed, changeStart);
        compressed.push_back(static_cast<uint8_t>(deltas.size()));
        for (int8_t delta : deltas) compressed.push_back(static_cast<uint8_t>(delta));
    }
    return compressed;
}

void VideoCompressor::writeVarint(std::vector<uint8_t>& out, size_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value));
}

double VideoCompressor::MP4FileSize(const std::string& filePath) {
    try {
        auto sizeBytes = std::filesystem::file_size(std::filesystem::path(filePath));
        return static_cast<double>(sizeBytes) / (1024.0 * 1024.0);
    } catch (...) { return -1.0; }
}

void VideoCompressor::printStats() {
    std::cout << "\n=== COMPRESSION DIAGNOSTICS ===" << std::endl;
    std::cout << "Total frames: " << savedFrameCount << std::endl;
    std::cout << "Key frames: " << keyframeCount << " ("
        << totalKeyframeBytes / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "Delta frames: " << (savedFrameCount - keyframeCount)
        << " (" << totalDeltaBytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    std::cout << "\nCompressed video saved to: " << outputFile << std::endl;

    uint64_t pixels = static_cast<uint64_t>(metadata.width) * metadata.height;
    uint64_t originalSize = (pixels * 3 * totalProcessedFrames) / 2; // YUV420P ratio
    
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
