#include <thread>
#include <chrono>
#include "beamforming_mic_array.h"

const int SR=48000;
const int BUFFER_SIZE=384;
int main(){
    audio_config_t layout(BUFFER_SIZE, 6, 2, SR);
    layout.positions[0]=Pointf(-0.0232,0,0.0401);
    layout.positions[1]=Pointf(-0.0463,0,0);
    layout.positions[2]=Pointf(-0.0232,0,-0.0401);
    layout.positions[3]=Pointf(0.0232,0,-0.0401);
    layout.positions[4]=Pointf(0.0463,0,0);
    layout.positions[5]=Pointf(0.0232,0,0.0401);

    beamforming_mic_array array(layout);
    array.init_mic_array(1);
    array.start();
    
    std::this_thread::sleep_for(std::chrono::seconds(30));
    array.stop();
}
