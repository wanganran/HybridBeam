#include "beamform.h"

const float EPS=0.000001;
const float MAX_DIST=1000;
const int SUBBAND=3;
static webrtc::SphericalPointf get_dir(float angle){
    return webrtc::SphericalPointf(angle, 0, MAX_DIST);
}

bool webrtc_beamform::init_beamform(float initial_angle, const audio_config_t& layout){
    int N=layout.num_mic;
    int buffer_size=layout.buffer_size;
    std::vector<webrtc::Point> mics(N);
    for(int i=0;i<N;i++) mics[i]=webrtc::Point(layout.positions[i].x, layout.positions[i].y, layout.positions[i].z);

    beamformer=new webrtc::NonlinearBeamformer(mics);

    beamformer->InitializeF(buffer_size/SUBBAND, layout.sampling_rate/SUBBAND);
    
    auto dir=get_dir(initial_angle);
    beamformer->AimAt(dir);

    input_3b=new webrtc::ChannelBuffer<float>(buffer_size, N, SUBBAND);

    output_3b=new webrtc::ChannelBuffer<float>(buffer_size, layout.num_out, SUBBAND);

    input_1b=new webrtc::ChannelBuffer<float>(buffer_size, N);

    output_1b=new webrtc::ChannelBuffer<float>(buffer_size, layout.num_out);

    filterbank.clear();
    for(int i=0;i<N;i++)
        filterbank.push_back(webrtc::ThreeBandFilterBank(buffer_size));
    return true;
}


bool webrtc_beamform::process_beamform(float* input, float* output){
    int channel=input_1b->num_channels();
    int buffer_size=input_1b->num_frames();

    for(int i=0;i<channel;i++){
        float* dest=input_1b->bands(i)[0];
        for(int j=0;j<buffer_size;j++)
            dest[j]=input[j*channel+i];
        filterbank[i].Analysis(dest, buffer_size, input_3b->bands(i));
    }
    beamformer->ProcessChunk(*input_3b, output_3b);

    int out_channel=output_1b->num_channels();
    

    filterbank[0].Synthesis(output_3b->bands(0), buffer_size/SUBBAND, output_1b->bands(0)[0]);
    for(int i=0;i<out_channel;i++){
        float* dest=output_1b->bands(0)[0];
        for(int j=0;j<buffer_size;j++)
            output[j*out_channel+i]=dest[j];
    }
    return true;
}
bool webrtc_beamform::adjust_direction(float new_angle){
    beamformer->AimAt(get_dir(new_angle));
    return true;
}

void webrtc_beamform::stop_beamform(){
    delete input_1b;
    delete output_1b;
    delete input_3b;
    delete output_3b;
    delete beamformer;
}
