#include <thread>
#include <portaudio.h>
#include <math.h>
#include <string.h>
#include "mic_array.h"

int mic_array::paCallback(const void* inputBuffer, void* outputBuffer, 
    unsigned long frames,const PaStreamCallbackTimeInfo* timeinfo,
    PaStreamCallbackFlags statusFlags,
    void* userData){

    mic_array* context=(mic_array*)userData;
    auto config=context->config;
    assert(frames==(unsigned long)(config.buffer_size));

    float* copy=new float[config.num_mic*frames];
    memcpy(copy, inputBuffer, sizeof(float)*config.num_mic*frames);
    context->queue_in.enqueue(copy);
    if(context->output_wait>0){
        context->output_wait--;
        memset(outputBuffer, 0, sizeof(float)*config.num_out*frames);
    } else {
        float* output=context->get_latest_output();
        if(output==NULL){
            memset(outputBuffer, 0, sizeof(float)*config.num_out*frames);
        } else {
            //printf("%f %f %f %f\n", output[0], output[1], output[config.num_out*frames-4], output[config.num_out*frames-3]);
            memcpy(outputBuffer, output, sizeof(float)*config.num_out*frames);
            delete[] output;
        }
    }
    return 0;
}

void mic_array::worker_func(){
    float* result=NULL;

    while(!worker_stop){
        if(queue_in.try_dequeue(result)){
            float* output=new float[config.buffer_size*config.num_out];
            process(result, output, config.buffer_size);
            queue_out.enqueue(output);
            delete[] result;
        } else std::this_thread::yield();
    }

    while(queue_in.try_dequeue(result))
        delete[] result;
}

bool mic_array::init_mic_array(int output_wait){
    PaStreamParameters inputPara, outputPara;
    PaStream *stream=NULL;

    PaError err;

    err=Pa_Initialize();
    if(err!=paNoError) return false;
    
    inputPara.device=Pa_GetDefaultInputDevice();
    outputPara.device=Pa_GetDefaultOutputDevice();
    auto inputInfo=Pa_GetDeviceInfo(Pa_GetDefaultInputDevice());
    auto outputInfo=Pa_GetDeviceInfo(Pa_GetDefaultOutputDevice());
    inputPara.channelCount=config.num_mic;
    inputPara.sampleFormat=paFloat32;
    inputPara.suggestedLatency=inputInfo->defaultLowInputLatency;
    inputPara.hostApiSpecificStreamInfo=NULL;
    outputPara.channelCount=config.num_out;
    outputPara.sampleFormat=paFloat32;
    outputPara.suggestedLatency=outputInfo->defaultLowOutputLatency;
    outputPara.hostApiSpecificStreamInfo=NULL;

    err=Pa_OpenStream(
        &stream,
        &inputPara,
        &outputPara,
        config.sampling_rate,
        config.buffer_size,
        paClipOff,
        paCallback,
        this);

    if(err!=paNoError) return false;
    this->stream=stream;

    this->output_wait=output_wait;
    return true;
}

float* mic_array::get_latest_output(){
    float* result=NULL;
    if(queue_out.try_dequeue(result))
        return result;
    else return NULL;
}

bool mic_array::start(){
    auto err=Pa_StartStream(stream);
    if(err!=paNoError)
        return err;
    working_thread=std::thread(&mic_array::worker_func, this);
    return true;
}

bool mic_array::stop(){
    auto err=Pa_StopStream(stream);
    if(err==paNoError){
        worker_stop=true;
        working_thread.join();
        return true;
    } else {
        //TODO
        return false;
    }
}
