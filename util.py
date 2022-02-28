import numpy as np

from scipy import signal
import torch
import torch.nn.functional as F
import subprocess

EPS=1e-8

def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

def sdr(estimated, original):
    target=pow_p_norm(original)
    noise=pow_p_norm(estimated-original)
    
    sdr = 10 * torch.log10(target / (noise + EPS) + EPS)
    return sdr.squeeze_(dim=-1)


def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)

# x: total; y_pred: output; y_true: gt
def wsdr_loss(x, y_pred, y_true):
    # x: [B, 1, L]

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=-1)
        den = torch.norm(true, p=2, dim=-1) * torch.norm(pred, p=2, dim=-1)
        return -(num / (den + EPS))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=-1) / (torch.sum(y_true**2, dim=-1) + torch.sum(z_true**2, dim=-1) + EPS)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation
    Input:
        estimated_signal and reference signals are (N,) numpy arrays
    Returns: SI-SDR as scalar
    """
    # first align them
    '''
    lag_values = np.arange(-len(estimated_signal)+1, len(estimated_signal))
    crosscorr = signal.correlate(estimated_signal, reference_signals)
    max_crosscorr_idx = np.argmax(crosscorr)
    lag = -lag_values[max_crosscorr_idx]
    if lag>0:
        estimated_signal=estimated_signal[:-lag]
        reference_signals=reference_signals[lag:]
    else:
        estimated_signal=estimated_signal[-lag:]
        reference_signals=reference_signals[:lag]

    plt.figure()
    plt.plot(estimated_signal)
    plt.plot(reference_signals)
    plt.show()
    '''
    
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals
    
    
    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * np.log10(Sss/Snn)

    return SDR

def real_bpf(sr, fmin, fmax, fw):
    fw_half=fw//2
    wind=np.zeros((fw,), dtype='complex')
    for i in range(fw):
        j=i-fw_half
        if j!=0:
            wind[i]=np.sin(2*np.pi*j*fmax/sr)/j/np.pi-np.sin(2*np.pi*j*fmin/sr)/j/np.pi
    wind[fw_half]=2*(fmax-fmin)/sr
    s=np.sum(wind)/fw
    wind=wind-s
    return wind

def calculate_gain(original, mixed, beamformed, fs=24000, freq_low=50, freq_high=11000, wind_size=129):
    # first high pass filter
    if wind_size is not None:
        wind=real_bpf(fs, freq_low, freq_high, wind_size)
        original=np.real(np.convolve(original, wind, 'valid'))
        mixed=np.real(np.convolve(mixed, wind, 'valid'))
        beamformed=np.real(np.convolve(beamformed, wind, 'valid'))
    
    l=np.min([len(original), len(mixed), len(beamformed)])
    si_sdr1=si_sdr(mixed[:l], original[:l])
    si_sdr2=si_sdr(beamformed[:l], original[:l])
    return (si_sdr2-si_sdr1, si_sdr1, si_sdr2)

def db(x):
    return 10*np.log10(x)

def dbto(x):
    return 10**(x/10)

def power(signal):
    return np.sum(np.abs(signal)**2)/signal.size

def normalize(signal): # keep a std of 0.05 and min/max of +-1
    return np.clip(signal/np.std(signal)*0.05, -1, 1)

def mix(signal, noise, target_snr_db):
    psig=power(signal)
    pnoi=power(noise)
    newnoise=noise*np.sqrt(dbto(db(psig/pnoi)-target_snr_db))
    res=signal+newnoise
    return res, newnoise

def delay(signal, sample):
    t=-sample/len(signal)
    sigf=np.fft.fft(signal)
    sigf[len(signal)//2]*=np.exp(1j*2*np.pi*len(signal)/2*t)
    for i in range(1, len(signal)//2):
        sigf[i]*=np.exp(1j*2*np.pi*i*t)
        sigf[len(signal)-i]*=np.exp(-1j*2*np.pi*i*t)
    return np.fft.ifft(sigf)

def alignChannel(signal, angle, mic_array_layout, V=343.0, fs=48000):
    # signal is [N, C]
    n_mic=signal.shape[1]
    result=np.zeros(signal.shape)
    
    for m in range(n_mic):
        dx=mic_array_layout[0, m]
        dy=mic_array_layout[1, m]
        d=dx*np.cos(angle)+dy*np.sin(angle)
        d=d/V*fs
        result[:,m]=delay(signal[:,m], d)
    return result

def randomEqualize(wav, maxdb, mindb, points):
    eqpts=np.random.random(points*2)*(maxdb-mindb)+mindb
    eqpts=10**(eqpts/20)
    eqpts[points+1:]=eqpts[points-1:0:-1]
    eqpts=np.fft.ifft(eqpts)[:points]
    
    return np.convolve(wav, np.real(eqpts), 'same')