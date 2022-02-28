#ifndef BEAMFORM_MIC_ARRAY_H
#define BEAMFORM_MIC_ARRAY_H

#include "mic_array.h"
#include "beamform.h"

#define PI 3.1415926

class beamforming_mic_array:public mic_array{
private:
    webrtc_beamform beamform;
public:
    beamforming_mic_array(audio_config_t config):mic_array(config){}
    virtual bool init_mic_array(int output_wait){
        mic_array::init_mic_array(output_wait);
        beamform.init_beamform(PI/2, config);
        return true;
    }
    virtual bool stop(){
        mic_array::stop();
        beamform.stop_beamform();
        return true;
    }
protected:
    virtual void process(float* input, float* output, int frame){
        beamform.process_beamform(input, output);
    }
};

#endif