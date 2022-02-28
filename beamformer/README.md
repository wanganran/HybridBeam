# webrtc beamformer running on ReSpeaker
- first clone https://github.com/wanganran/webrtc-audio-processing and build it using the below procedures:
  - install meson: https://github.com/mesonbuild/meson
  - install ninja: https://github.com/ninja-build/ninja
  - goto build/ folder, and type: /folder/to/meson.py ../
  - then type: ninja
- run `cmake .` and `make` to build and `beamform_mic_array` to run
