//
// Created by Anran Wang on 2019-08-22.
//

#ifndef SOUND_CPP_BPF_H
#define SOUND_CPP_BPF_H

#include <vector>
#include "types.h"
/**
 * band pass filter utility
 */
class bpf
{
private:
    std::vector<float_t> mask;
    std::vector<float_t> window;
    int offset;

public:
    bpf(int halfwind, float_t sr, float_t from, float_t to);
    void input_arr(float_t* signal, int len){
    	for(int i=0;i<len;i++)
		signal[i]=input(signal[i]);
    	
    }
    float_t input(float_t signal);
    void reset();
};

#endif //SOUND_CPP_BPF_H