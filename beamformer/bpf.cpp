//
// Created by Anran Wang on 2019-08-22.
//

#include <math.h>
#include "bpf.h"

bpf::bpf(int halfwind, float_t sr, float_t from, float_t to) : mask(2 * halfwind + 1), window(2 * halfwind + 1), offset(0)
{
    int size=2*halfwind+1;
    for (int i = 0; i < size; i++)
    {
        mask[i] = 0.53836 - 0.46164 * cos(2 * PI * i / (2 * halfwind));
        auto n = halfwind - i;
        mask[i] = mask[i] * (sin(2 * PI * n * to / sr) / n / PI -
                             sin(2 * PI * n * from / sr) / n / PI);
    }
    mask[halfwind] = 2 * (to - from) / sr;
    float_t sum = 0;
    for (int i = 0; i < size; i++)
        sum += mask[i];
    for (int i = 0; i < size; i++)
        mask[i] *= size;
}

float_t bpf::input(float_t signal)
{
    int size=mask.size();
    window[offset % size] = signal;
    float_t sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += window[(i + offset + 1) % size] * mask[i];
    }
    offset += 1;
    return sum/size;
}

void bpf::reset()
{
    for (int i = 0; i < (int)window.size(); i++)
        window[i] = 0;
}