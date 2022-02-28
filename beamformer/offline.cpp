#include <iostream>
#include <fstream>
#include <vector>
#include <sndfile.h>
#include "beamform.h"

using namespace std;
const int MAX_CHANNEL=20;
const int BUFFER_SIZE=240;
int main(int argc, const char* argv[]){
    if(argc<=4){
        cout<<"Argument is needed"<<endl;
	cout<<"Parameters: [multi-channel audio file] [angle in degree] [mic array layout text file] [output path] [buffer size in sample (optional)]"<<endl;
    }
    const char* path=argv[1];
    float angle=atof(argv[2]);
    const char* layout_path=argv[3];
    const char* out_path=argv[4];
    int buffer_size=(argc>=6?atoi(argv[5]):BUFFER_SIZE);
    SF_INFO in_file_info;
    SNDFILE* in_file=sf_open(path, SFM_READ, &in_file_info);
    int channel=in_file_info.channels;
    int length=in_file_info.frames;
    int sr=in_file_info.samplerate;
    std::vector<std::vector<float>> signal(channel);
    
    float _buffer[MAX_CHANNEL];
    for(int i=0;i<length;i++){
        sf_read_float(in_file, _buffer, channel);
        for(int i=0;i<channel;i++){
            signal[i].push_back(_buffer[i]);
        }
    }
    sf_close(in_file);

    //beamforming
    webrtc_beamform beamform;
    audio_config_t layout(buffer_size, channel, 1, sr);
    
    std::ifstream in_profile(layout_path);
    for(int i=0;i<channel;i++){
        float x,y;
        in_profile>>x>>y;
        layout.positions[i]=Pointf(x,y,0);
    }
    in_profile.close();
    float rad=angle/180.0*3.141592653;
    cout<<rad<<endl;
    beamform.init_beamform(rad, layout);
    beamform.adjust_direction(rad);
    std::vector<float> buffer(channel*buffer_size);
    std::vector<float> buffer_out(buffer_size);
    std::vector<float> signal_out;
    for(int i=0;i<length;i+=buffer_size){
        for(int j=0;j<buffer_size;j++){
            for(int k=0;k<channel;k++){
                buffer[j*channel+k]=signal[k][j+i];
            }
        }
        beamform.process_beamform(&(buffer[0]), &(buffer_out[0]));
        for(int j=0;j<buffer_size;j++)
            signal_out.push_back(buffer_out[j]);
    }
    SF_INFO out_file_info;
    out_file_info.channels=1;
    out_file_info.samplerate=sr;
    out_file_info.format=SF_FORMAT_WAV|SF_FORMAT_PCM_16;

    SNDFILE* out_file=sf_open(out_path, SFM_WRITE, &out_file_info);
    for(int i=0;i<signal_out.size();i+=buffer_size){
        sf_write_float(out_file, &(signal_out[i]), buffer_size);
    }
    sf_write_sync(out_file);
    sf_close(out_file);
    
}
