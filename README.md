# Video Delta Compressor (.ige)

A specialized C++ video compression tool designed for high-efficiency storage of static or low-motion footage. It uses **Delta Encoding** and **Run-Length Encoding (RLE)** to store only the pixel changes between frames, rather than full images.

## Key Features

*   **Delta Encoding**: Stores only the difference between frames. If a pixel doesn't change, it takes up 0 space.
*   **Run-Length Encoding (RLE)**: Compresses sequences of identical pixels (e.g., solid backgrounds) extremely efficiently.
*   **CUDA Hardware Acceleration**: Uses NVIDIA GPUs to decode the input video significantly faster.
*   **Resolution Scaling**: Built-in downscaling to massively reduce file size.
*   **Bit Quantization**: Reduces color precision to remove camera noise and improve compression ratios.
*   **Variable Frame Rate**: Configurable frame sampling interval (e.g., store only every 10th frame).

## Ideal Use Cases

This format (`.ige`) is **not** designed to replace H.264/MP4 for action movies. It excels in specific scenarios:

1.  **Security & Surveillance**: Efficiently stores hours of footage where nothing moves.
2.  **Screen Recording**: highly effective for desktop/software tutorials with large static UI elements.
3.  **Low-Power Playback**: The decoding algorithm is mathematically trivial (simple addition), allowing playback on very weak hardware that cannot decode MP4.
4.  **Glitch Art**: The raw delta format allows for easy datamoshing and visual effects.

## Usage

```powershell
video-compression.exe <input.mp4> <output.ige> [OPTIONS]
```

### Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--cuda` | Enable NVIDIA CUDA hardware acceleration | Off |
| `--width <W>` | Resize video to target width | Original |
| `--height <H>` | Resize video to target height | Original |
| `--quantize <0-4>` | Bit-shift color reduction (Higher = Smaller File) | 0 |
| `--interval <N>` | Process every Nth frame (Frame skipping) | 10 |
| `--threshold <0-255>` | Pixel change sensitivity (Higher = Ignore small changes) | 15 |
| `--keyframe <N>` | Insert a full keyframe every N saved frames | 30 |

### Examples

**High Compression (Surveillance/Archival):**
Downscale to 360p, reduce color precision, and ignore minor lighting changes.
```powershell
video-compression.exe input.mp4 secure.ige --width 640 --height 360 --quantize 3 --threshold 40 --interval 10
```

**High Quality (Screen Recording):**
Keep original resolution and colors, capture every frame.
```powershell
video-compression.exe input.mp4 screen.ige --interval 1 --quantize 0 --threshold 5
```

## Building

**Requirements:**
*   CMake 3.10+
*   C++17 Compiler (MSVC recommended on Windows)
*   FFmpeg Libraries (Dev packages)

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
