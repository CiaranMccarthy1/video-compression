# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("video-compression")

set(CMAKE_CXX_STANDARD 17)

# Set FFmpeg path
set(FFMPEG_DIR "C:/ffmpeg")

# Include directories for FFmpeg headers
include_directories(${FFMPEG_DIR}/include)

# Link directories for FFmpeg libraries
link_directories(${FFMPEG_DIR}/lib)

# Create executable
add_executable(video-compression main.cpp)

# Link FFmpeg libraries
target_link_libraries(video-compression
        avcodec
        avformat
        avutil
        swscale
)
