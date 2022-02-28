#ifndef MIC_ARRAY_H
#define MIC_ARRAY_H

#include <thread>
#include <portaudio.h>
#include <vector>
#include "readerwriterqueue.h"

struct Pointf{
    float x,y,z;
    Pointf(float _x, float _y, float _z):x(_x), y(_y), z(_z){}
    Pointf():x(0),y(0),z(0){}
};

struct audio_config_t{
    int buffer_size;
    int sampling_rate;
    int num_mic;
    std::vector<Pointf> positions;
    int num_out;
    audio_config_t(int buffer_size, int n, int out, int sr):buffer_size(buffer_size), sampling_rate(sr), num_mic(n), positions(n), num_out(out){}
};

class mic_array{
protected:
    int output_wait;
    audio_config_t config;
    PaStream* stream;

    std::thread working_thread;
    bool worker_stop;
    moodycamel::ReaderWriterQueue<float*> queue_in;
    moodycamel::ReaderWriterQueue<float*> queue_out;

    void worker_func();

    static int paCallback(const void* inputBuffer, void* outputBuffer, 
        unsigned long frames,const PaStreamCallbackTimeInfo* timeinfo,
        PaStreamCallbackFlags statusFlags,
        void* userData);
public:
    mic_array(audio_config_t config):output_wait(0), config(config), stream(NULL), worker_stop(false){}
    virtual bool init_mic_array(int output_wait);

    virtual bool start();

    float* get_latest_output();

    virtual bool stop();
    ~mic_array(){
        if(stream){
            Pa_AbortStream(stream);
            Pa_CloseStream(stream);
        }
        Pa_Terminate();

        float* out=NULL;
        while((out=get_latest_output())!=NULL)delete[] out;
    }
protected:
    virtual void process(float* input, float* output, int frames)=0;
};

#endif