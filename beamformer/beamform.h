#ifndef BEAMFORM_H
#define BEAMFORM_H

#include <vector>
#define WEBRTC_AUDIO_PROCESSING_ONLY_BUILD
#include <webrtc/modules/audio_processing/beamformer/nonlinear_beamformer.h>
#include <webrtc/modules/audio_processing/three_band_filter_bank.h>
#include "mic_array.h"

class beamform{
public:
    virtual bool init_beamform(float initial_angle, const audio_config_t& layout)=0;

    virtual bool process_beamform(float* input, float* output)=0;

    virtual bool adjust_direction(float new_angle)=0;

    virtual void stop_beamform()=0;
};

class webrtc_beamform:public beamform{
private:
    webrtc::NonlinearBeamformer* beamformer;
    std::vector<webrtc::ThreeBandFilterBank> filterbank;
    webrtc::ChannelBuffer<float>* input_1b;
    webrtc::ChannelBuffer<float>* output_1b;
    webrtc::ChannelBuffer<float>* input_3b;
    webrtc::ChannelBuffer<float>* output_3b;
public:

    webrtc_beamform():beamformer(NULL), input_1b(NULL), output_1b(NULL), input_3b(NULL), output_3b(NULL){}
    virtual bool init_beamform(float initial_angle, const audio_config_t& layout);

    virtual bool process_beamform(float* input, float* output);

    virtual bool adjust_direction(float new_angle);

    virtual void stop_beamform();
};

#endif