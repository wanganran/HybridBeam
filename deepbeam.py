import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import asteroid

class ModTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, eps=1e-7):
        x_re, x_im = x[0], x[1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2 + eps)
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = torch.tanh(norm)
        return (activated_norm * phase_re, activated_norm * phase_im)
        
    
class TReLU(nn.Module):
    def __init__(self, input_shape, bias=False):
        super().__init__()
        self.transform = torch.nn.Parameter(torch.randn((2,2)+input_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn((2,)+input_shape))
        self.relu = torch.nn.LeakyReLU()
        self.has_bias=bias
        
    def forward(self, tensor):
        real=tensor[0]
        imag=tensor[1]
        
        t00=self.transform[0,0]
        t01=self.transform[0,1]
        t10=self.transform[1,0]
        t11=self.transform[1,1]
        if self.has_bias:
            b0=self.bias[0]
            b1=self.bias[1]

            newreal=real*t00+imag*t01+b0
            newimag=real*t10+imag*t11+b1
        else:
            newreal=real*t00+imag*t01
            newimag=real*t10+imag*t11
            
        return (self.relu(newreal),
            self.relu(newimag))

def cMul(t1,t2):
    real=t1[0]*t2[0]-t1[1]*t2[1]
    imag=t1[1]*t2[0]+t1[0]*t2[1]
    return (real,imag)

class ComplexSTFTWrapper(nn.Module):
    def __init__(self, win_length, hop_length, center=True):
        super(ComplexSTFTWrapper,self).__init__()
        self.win_length=win_length
        self.hop_length=hop_length
        self.center=center
        
    def transform(self, input_data):
        B,C,L=input_data.shape
        input_data=input_data.view(B*C, L)
        r=torch.stft(input_data, n_fft=self.win_length, hop_length=self.hop_length, center=self.center, return_complex=False)
        _,F,T,_=r.shape
        r=r.view(B,C,F,T,2)
        return (r[...,0], r[..., 1])
                              
    def reverse(self, input_data):
        r,i=input_data
        B,C,F,T=r.shape
        r=r.flatten(0,1)
        i=i.flatten(0,1)
        input_data=torch.stack([r,i], dim=-1)
        r=torch.istft(input_data, n_fft=self.win_length, hop_length=self.hop_length, center=self.center, return_complex=False) # B, L
        return r.view(B,C,-1)
        
    def forward(self, x):
        return self.reverse(self.transform(x))

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=(0,0), dilation=1, groups=1, bias=True, complex_mul=True, causal=True):
        super(ComplexConv2d,self).__init__()
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.complex_mul=complex_mul
        self.causal=causal
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        if self.complex_mul:
            real = self.conv_re(x[0]) - self.conv_im(x[1])
            imag = self.conv_re(x[1]) + self.conv_im(x[0])
        else:
            real = self.conv_re(x[0])
            imag = self.conv_im(x[1])
        
        if self.causal and self.padding[1]>0:
            real=real[..., :-self.padding[1]]
            imag=imag[..., :-self.padding[1]]

        output = (real,imag)
        return output
    
class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, causal=True):
        super(ComplexConv1d,self).__init__()
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.causal=causal
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[0]) - self.conv_im(x[1])
        imag = self.conv_re(x[1]) + self.conv_im(x[0])
        if self.causal and self.padding>0:
            real=real[..., :-self.padding]
            imag=imag[..., :-self.padding]
        output = (real,imag)
        return output
    
class CBatchNorm(nn.Module):
    def __init__(self, ch, momentum, is_2d, affine=True):
        super().__init__()
        if is_2d:
            self.norm=nn.BatchNorm2d(ch, momentum, affine)
        else:
            self.norm=nn.BatchNorm1d(ch, momentum, affine)
            
    def forward(self, t):
        (ta, tb)=t
        B=ta.shape[0]
        tn=self.norm(torch.cat((ta,tb), dim=0))
        t=tn.split(B)
        return t
    
def tuple_plus(a,b):
    return (a[0]+b[0], a[1]+b[1])

class ComplexTCN(nn.Module):
    def __init__(self, ch_mid, ch_hid, kernel, dilation, shortcut=False):
        super().__init__()
        self.conv1=ComplexConv1d(ch_mid, ch_hid, kernel, dilation=dilation, padding=dilation*(kernel-1))
        self.act=TReLU((ch_hid, 1))
        self.norm=CBatchNorm(ch_hid, 0.02, False)
        self.conv2=ComplexConv1d(ch_hid, ch_mid, 1)
        
        self.shortcut=shortcut
        if shortcut:
            self.conv3=ComplexConv1d(ch_hid, ch_mid, 1)
        
    def forward(self, t):
        tori=t
        t=self.conv1(t)
        t=self.act(t)
        t=self.norm(t)
        if self.shortcut:
            t2=self.conv3(t)
        else:
            t2=None
        t=self.conv2(t)
        
        return (tuple_plus(t,tori), t2)

class BeamformerModel(nn.Module):
    def __init__(self, ch_in, synth_mid, synth_hid, block_size, kernel, synth_layer, synth_rep, lookahead=0):
        super().__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size*2)
        self.ch_in=ch_in
        self.kernel=kernel
        self.ch=256
        
        self.bn_first=CBatchNorm(synth_mid, 0.02, False)        
        
        self.synth_first=ComplexConv1d(self.ch, synth_mid, 1)
        
        self.synth_final=nn.Sequential(
            ComplexConv1d(synth_mid, self.ch, 1),
            TReLU((self.ch, 1)),
            ComplexConv1d(self.ch, self.ch, 1))
        
        self.synth_last=ComplexConv1d(self.ch, block_size+1, 1)
        
        self.synth_act=ModTanh()
        self.synth_conv=nn.ModuleList()
        
        self.reduce_conv=nn.ModuleList()
        
        for r in range(synth_rep):
            dilation=1
            if r!=0:
                self.reduce_conv.append(ComplexConv1d(synth_mid, synth_mid, 2, stride=2))
                
            for i in range(synth_layer):
                self.synth_conv.append(ComplexTCN(synth_mid, synth_hid, kernel, dilation, i==synth_layer-1))
                dilation*=kernel
        self.preconv=ComplexConv1d(ch_in*(block_size+1), self.ch, 1)
        
        self.lookahead=lookahead//(block_size//2)
        self.synth_layer=synth_layer
        
        
        
    def forward(self, wav): # wav: B, C, T
        spec=self.stft.transform(wav) #B,C,F,T
        
        B,Cin,_,T=spec[0].shape
        
        tlr=spec[0].flatten(1,2)
        tli=spec[1].flatten(1,2)
        
        tl=self.preconv((tlr,tli))
        
        t=tl
        
        tl=self.synth_first(tl)
        tl=self.bn_first(tl)
        
        ts_sum=None
        scale_factor=1
        for i in range(len(self.synth_conv)):
            if i%self.synth_layer==0 and i!=0:
                tl=self.reduce_conv[i//self.synth_layer-1](tl)
                scale_factor*=2
                
            tl, ts=self.synth_conv[i](tl)
            if ts is not None:
                if ts_sum is None:
                    ts_sum=ts
                else:
                    ts0=F.upsample_nearest(ts[0], scale_factor=scale_factor)
                    ts1=F.upsample_nearest(ts[1], scale_factor=scale_factor)
                    
                    ts=ts0,ts1
                    ts_sum=tuple_plus(ts_sum, ts)
            
        spec=self.synth_final(ts_sum) # B, freq*ch_in, T
        
        padding=spec[0][..., :self.lookahead], spec[1][..., :self.lookahead]
        padding=padding[0]-padding[0], padding[1]-padding[1]
        
        spec=spec[0][..., self.lookahead:], spec[1][..., self.lookahead:]
        spec=torch.cat([spec[0], padding[0]], dim=-1), torch.cat([spec[1], padding[1]], dim=-1)
        
        spec=cMul(self.synth_act(spec), t)
        spec=self.synth_last(spec)
        spec=spec[0].unsqueeze(1), spec[1].unsqueeze(1)
        return self.stft.reverse(spec)
        