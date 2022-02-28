cmake_minimum_required(VERSION 3.10)
project(beamform_mic_array)

set(CMAKE_CXX_STANDARD 17)
link_directories("/usr/local/lib")
include_directories("/usr/local/include")
include_directories("../webrtc-audio-processing")

add_executable(beamform_mic_array offline.cpp webrtc_beamform.cpp mic_array.cpp)

SET(CMAKE_CXX_FLAGS "-DASIO_STANDALONE -Wall -O3")
target_link_libraries(beamform_mic_array pthread portaudio sndfile ../webrtc-audio-processing/build/webrtc/modules/audio_processing/libwebrtc_audio_processing.so ../webrtc-audio-processing/build/webrtc/libwebrtc.a)
