# HybridBeam: Hybrid Neural Networks for On-Device Directional Hearing

This is the source code for our AAAI 22 paper: Hybrid Neural Networks for On-Device Directional Hearing.

To generate the synthetic datasets, you need to first download the original speech and noise datasets by running `download_datasets.sh` and run the `generate_datasets.py` using Python 3. The network is defined in `deepbeam.py` and can be trained using `train.py`. We also attached a pretrained model for 6-mic DeepBeam+ model in pretrained `pretrained_6mic.bin`, where you can use to process existing data using `inference.py`.

The code requires a C/C++ implementation of WebRTC beamformer. The code is in the `beamformer` folder where you can compile following the `README.md` there. and a compiled x86-64 version is `beamform_mic_array` where you can directly call using the Python code.