import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy import signal as sg
import numpy.matlib as npm
import os
import uuid

def stab(mat, theta, num_channels):
    d = np.power(np.array(10, dtype=np.complex64) , np.arange( - num_channels, 0, dtype=np.float))
    result_mat = mat
    for i in range(1, num_channels + 1):
        if np.linalg.cond(mat) > theta:
            return result_mat
        result_mat = result_mat + d[i - 1] * np.eye(num_channels, dtype=np.complex64)
    return result_mat

def get_3dim_spectrum(wav_name, channel_vec, start_point, stop_point, frame, shift, fftl):
    """
    dump_wav : channel_size * speech_size (2dim)
    """
    samples, _ = sf.read(wav_name.replace('{}', str(channel_vec[0])), start=start_point, stop=stop_point, dtype='float32')
    if len(samples) == 0:
        return None,None
    dump_wav = np.zeros((len(channel_vec), len(samples)), dtype=np.float16)
    dump_wav[0, :] = samples.T
    for ii in range(0,len(channel_vec) - 1):
        samples,_ = sf.read(wav_name.replace('{}', str(channel_vec[ii +1 ])), start=start_point, stop=stop_point, dtype='float32')
        dump_wav[ii + 1, :] = samples.T    

    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.hanning(fftl + 1, 'periodic')[: - 1]
    multi_window = npm.repmat(window, len(channel_vec), 1)    
    st = 0
    ed = frame
    number_of_frame = np.int((len(samples) - frame) /  shift)
    spectrums = np.zeros((len(channel_vec), number_of_frame, np.int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fftl, axis=1)[:, 0:np.int(fftl / 2) + 1] # channel * number_of_bin        
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums, len(samples)

def get_3dim_spectrum_from_data(wav_data, frame, shift, fftl):
    """
    dump_wav : channel_size * speech_size (2dim)
    """
    len_sample, len_channel_vec = np.shape(wav_data)            
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.hanning(fftl + 1, 'periodic')[: - 1]
    multi_window = npm.repmat(window, len_channel_vec, 1)    
    st = 0
    ed = frame
    number_of_frame = np.int((len_sample - frame) /  shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, np.int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):       
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fftl, axis=1)[:, 0:np.int(fftl / 2) + 1] # channel * number_of_bin        
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums, len_sample

def my_det(matrix_):
    sign, lodget = np.linalg.slogdet(matrix_)
    return np.exp(lodget)

def spec2wav(spectrogram, sampling_frequency, fftl, frame_len, shift_len):
    n_of_frame, fft_half = np.shape(spectrogram)    
    hanning = sg.hanning(fftl + 1, 'periodic')[: - 1]    
    cut_data = np.zeros(fftl, dtype=np.complex64)
    result = np.zeros(sampling_frequency * 60 * 5, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]        
        cut_data[0:np.int(fftl / 2) + 1] = half_spec.T   
        cut_data[np.int(fftl / 2) + 1:] =  np.flip(np.conjugate(half_spec[1:np.int(fftl / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fftl))        
        result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2 * hanning.T) 
        start_point = start_point + shift_len
        end_point = end_point + shift_len
    return result[0:end_point - shift_len]

def multispec2wav(multi_spectrogram, beamformer, fftl, shift, multi_window, true_dur):
    channel, number_of_frame, fft_size = np.shape(multi_spectrogram)
    cut_data = np.zeros((channel, fftl), dtype=np.complex64)
    result = np.zeros((channel, true_dur), dtype=np.float32)
    start_p = 0
    end_p = start_p + fftl
    for ii in range(0, number_of_frame):
        cut_spec = multi_spectrogram[:, ii, :] * beamformer
        cut_data[:, 0:fft_size] = cut_spec
        cut_data[:, fft_size:] = np.transpose(np.flip(cut_spec[:, 1:fft_size - 1], axis=1).T)
        cut_data2 = np.real(ifft(cut_data, n=fftl, axis=1))
        result[:, start_p:end_p] = result[:, start_p:end_p] + (cut_data2 * multi_window)
        start_p = start_p + shift
        end_p = end_p + shift
    return np.sum(result[:,0:end_p - shift], axis=0)
           
        
def check_beamformer(freq_beamformer,theta_cov):
    freq_beamformer = np.real(freq_beamformer)
    if len(freq_beamformer[freq_beamformer>=theta_cov])!=0:
        return np.ones(np.shape(freq_beamformer),dtype=np.complex64) * (1+1j)
    return freq_beamformer


class minimum_variance_distortioless_response:
    
    def __init__(self,
                 n_mic,
                 sampling_frequency,
                 fft_length,
                 fft_shift,
                 sound_speed):    
        self.n_mic=n_mic
        self.sampling_frequency=sampling_frequency
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        self.sound_speed=sound_speed
    
    def normalize(self, steering_vector):
        for ii in range(0, self.fft_length):            
            weight = np.matmul(np.conjugate(steering_vector[:, ii]).T, steering_vector[:, ii])
            steering_vector[:, ii] = (steering_vector[:, ii] / weight) 
        return steering_vector        
    
    def get_spatial_correlation_matrix(self, multi_signal, use_number_of_frames_init=10):
        # multi_signal: L, C
        speech_length, number_of_channels = np.shape(multi_signal)
        if speech_length<self.fft_length: return None
        frequency_grid = np.arange(self.fft_length//2+1)*(self.sampling_frequency/self.fft_length)
        R_mean = np.zeros((self.n_mic, self.n_mic, len(frequency_grid)), dtype=np.complex64)
        
        multi_signal_cut = multi_signal[:self.fft_length, :]
        complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
        for f in range(0, len(frequency_grid)):
            R_mean[:, :, f] = np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
        
        return R_mean
    
    def get_mvdr_beamformer(self, steering_vector, R):
        frequency_grid = np.arange(self.fft_length//2+1)*(self.sampling_frequency/self.fft_length)    
        beamformer = np.ones((self.n_mic, len(frequency_grid)), dtype=np.complex64)
        for f in range(0, len(frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [self.n_mic, self.n_mic])
            try:
                inv_R = np.linalg.pinv(R_cut)
            except:
                return None
            a = np.matmul(np.conjugate(steering_vector[:, f]).T, inv_R)
            b = np.matmul(a, steering_vector[:, f])
            b = np.reshape(b, [1, 1])
            if b==0:
                return None
            beamformer[:, f] = np.matmul(inv_R, steering_vector[:, f]) / b # number_of_mic *1   = number_of_mic *1 vector/scalar        
        return beamformer
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)        

    def apply_beamformer_var(self, beamformer, complex_spectrum):
        # beamformer: [N, C, F]
        # complex_spectrum: [C, N, F]
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.sum(np.conjugate(beamformer[:, :, f]).T*complex_spectrum[:, :, f], axis=0)
        return spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)        
        

class WebrtcBeamformer:
    def __init__(self, buffer_size, mic_array_layout, sr, webrtc_path, temp_folder):
        self.webrtc_path=webrtc_path
        self.temp_folder=temp_folder
        assert(sr==48000)
        self.buffer_size=buffer_size
        self.n_mic=mic_array_layout.shape[1]
        #self.mic_array_layout=mic_array_layout
        self.mic_array_layout=mic_array_layout-np.tile(np.mean(mic_array_layout, axis=-1).reshape((3,1)), (1,self.n_mic))
        print(mic_array_layout)
        # output layout
        f=open(temp_folder+'/layout.txt', 'w')
        for i in range(mic_array_layout.shape[1]):
            for j in range(2):
                f.write("{0:.3f} ".format(self.mic_array_layout[j,i]))
        f.close()

    def __normalize(self, signal, maximum):
        return signal/np.max(np.abs(signal))*maximum
    
    def process(self, signal, angle):
        # angle is in degree
        tag=uuid.uuid4().hex
        path=self.temp_folder+'/original'+tag+'.wav'
        dest=self.temp_folder+'/output'+tag+'.wav'
        wavfile.write(path, 48000, signal)
        os.system(self.webrtc_path+' '+ path + ' ' + str(int(angle*180/np.pi)+180) + ' ' + self.temp_folder+'/layout.txt ' + dest + ' ' + str(self.buffer_size))
        result=wavfile.read(dest)[1][self.buffer_size//2:]/32768.0
        os.system('rm '+path)
        os.system('rm '+dest)
        return result
    
class WebrtcBeamformer16k:
    def __init__(self, buffer_size, mic_array_layout, sr, webrtc_path, temp_folder):
        self.webrtc_path=webrtc_path
        self.temp_folder=temp_folder
        assert(sr==16000)
        self.buffer_size=buffer_size
        self.n_mic=mic_array_layout.shape[1]
        self.mic_array_layout=mic_array_layout-np.tile(np.mean(mic_array_layout, axis=-1).reshape((3,1)), (1,self.n_mic))
        print(mic_array_layout)
        # output layout
        f=open(temp_folder+'/layout.txt', 'w')
        for i in range(mic_array_layout.shape[1]):
            for j in range(2):
                f.write("{0:.3f} ".format(self.mic_array_layout[j,i]))
        f.close()

    def __normalize(self, signal, maximum):
        return signal/np.max(np.abs(signal))*maximum
    
    def process(self, signal, angle):
        # angle is in degree
        tag=uuid.uuid4().hex
        path=self.temp_folder+'/original'+tag+'.wav'
        dest=self.temp_folder+'/output'+tag+'.wav'
        wavfile.write(path, 15999, signal)
        os.system(self.webrtc_path+' '+ path + ' ' + str(int(angle*180/np.pi)+180) + ' ' + self.temp_folder+'/layout.txt ' + dest + ' ' + str(self.buffer_size))
        result=wavfile.read(dest)[1][self.buffer_size//2:]/32768.0
        os.system('rm '+path)
        os.system('rm '+dest)
        return result
    
class OnlineMVDRBeamformer:
    def __init__(self, buffer_size, mic_array_layout, sr, V, adaptive=True, online=True, ds=False):
        # mic_array_layout: [3, C]
        self.buffer_size=buffer_size
        self.n_mic=mic_array_layout.shape[1]
        self.mic_array_layout=mic_array_layout-np.tile(np.mean(mic_array_layout, axis=-1).reshape((3,1)), (1, self.n_mic))
        self.sr=sr
        self.V=V
        self.adaptive=adaptive
        self.online=online
        
        if not adaptive:
            diag=0.05 if not ds else 1
            R=np.zeros((self.n_mic, self.n_mic, self.buffer_size//2+1))
            for i in range(self.n_mic):
                R[i,i,:]=1
            for i in range(self.n_mic):
                for j in range(i):
                    dist=self.__get_dist(i,j)
                    F=np.arange(self.buffer_size//2+1)*(self.sr/self.buffer_size)
                    R[i,j]=np.sinc(2*F*dist/V)
                    R[j,i]=R[i,j]

            self.corr=R*(1-diag)+np.tile(diag*np.identity(self.n_mic).reshape((self.n_mic, self.n_mic, 1)), (1,1,self.buffer_size//2+1))
        
    def __get_dist(self, i, j):
        return np.sqrt(np.sum((self.mic_array_layout[:, i]-self.mic_array_layout[:,j])**2))
    
        
    def __get_steering_vector(self, angle, sr, fft_len):
        frequency_vector=np.arange(fft_len)*sr/fft_len
        steering_vector=np.zeros((len(frequency_vector), self.n_mic), dtype='complex')

        # get delay
        delay=np.zeros((self.n_mic,))
        for m in range(self.n_mic):
            dx=self.mic_array_layout[0, m]
            dy=self.mic_array_layout[1, m]
            delay[m]=dx*np.cos(angle)+dy*np.sin(angle)

        
        for f, frequency in enumerate(frequency_vector):
            for m in range(self.n_mic):
                steering_vector[f,m]=np.exp(1j*2*np.pi*frequency*delay[m]/self.V)

        return steering_vector.T
    
    def __normalize(self, signal, maximum):
        return signal/np.max(np.abs(signal))*maximum
    
    def process(self, signal, angle):
        # signal: [N, C]
        step_size=self.buffer_size//2
        complex_spectrum,_=get_3dim_spectrum_from_data(signal, self.buffer_size, step_size, self.buffer_size)
        beamformer=minimum_variance_distortioless_response(self.n_mic, sampling_frequency=self.sr, \
                                             fft_length=self.buffer_size, fft_shift=step_size, sound_speed=self.V)
        length=signal.shape[0]
        v=self.__get_steering_vector(angle, self.sr, self.buffer_size)
        corr=None
        diag=0.05
        result=np.zeros((length,))
        spec_size=complex_spectrum.shape[1]
        beamformers=np.zeros((spec_size, self.n_mic, complex_spectrum.shape[2]), dtype=np.complex64)
        
        D=np.tile(diag*np.identity(self.n_mic).reshape((self.n_mic, self.n_mic, 1)), (1,1,self.buffer_size//2+1))
        if self.adaptive:
            for i in range(spec_size):
                newcorr=beamformer.get_spatial_correlation_matrix(signal[i*step_size:(i*step_size+self.buffer_size)])
                if self.online:
                    if corr is None:
                        corr=newcorr
                    else:
                        corr=corr+newcorr
                    if i%5==0:
                        beamformers[i]=beamformer.get_mvdr_beamformer(v, corr/(i+1)*(1-diag)+D)
                        if beamformers[i] is None or np.isnan(beamformers[i]).any() or np.isinf(beamformers[i]).any():
                            if i>0:
                                beamformers[i]=beamformers[i-1]
                            else:
                                beamformers[i]=v
                    else:
                        beamformers[i]=beamformers[i-1]
                        
                else:
                    if corr is None:
                        corr=newcorr
                    else:
                        corr+=newcorr
            if self.online:
                result=beamformer.apply_beamformer_var(beamformers, complex_spectrum)
            else:
                bf=beamformer.get_mvdr_beamformer(v, corr/spec_size)
                result=beamformer.apply_beamformer(bf, complex_spectrum)
        else:
            bf=beamformer.get_mvdr_beamformer(v, self.corr)
            result=beamformer.apply_beamformer(bf, complex_spectrum)
        return result
        
        