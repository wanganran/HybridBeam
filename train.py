# imports
import os
import numpy as np
import torchaudio
import torch
import asteroid
import librosa


cache_path='./temp'
fs=48000
V=343


vctk=torchaudio.datasets.VCTK('./')
from util import randomEqualize
# vctk adapter
class VCTKAudio:
    def __init__(self, vctk, numpy=True, eq=True, portion=None):
        self.vctk=vctk
        self.numpy=numpy
        self.eq=eq

        if portion is not None:
            self.start=portion[0]
            self.end=portion[1]
        else:
            self.start=0
            self.end=len(self.vctk)
        
    def __getitem__(self, idx):
        sig=self.vctk[idx+self.start][0][0].numpy() if self.numpy else self.vctk[idx+self.start][0][0]
        if self.eq:
            sig=randomEqualize(sig, 10, -10, 24)
        return sig
    
    def __len__(self):
        return self.end-self.start

class WHAM:
    def __init__(self, folder, eq=True):
        self.folder=folder
        self.noisy_files=[]
        for file in os.listdir(folder):
            self.noisy_files.append(os.path.join(folder, file))
        np.random.shuffle(self.noisy_files)
        self.eq=eq
        
    def __len__(self):
        return len(self.noisy_files)
    def __getitem__(self, idx):
        noise_audio, _ = librosa.load(self.noisy_files[idx], sr=fs, mono=True)
        if self.eq:
            noise_audio=randomEqualize(noise_audio, 10, -10, 24)
        return noise_audio

class FuseDataset:
    def __init__(self, *datasets):
        self.datasets=datasets
    
    def __len__(self):
        return sum([len(d) for d in self.datasets])
    
    def __getitem__(self, idx):
        for d in self.datasets:
            if len(d)>idx:
                return d[idx]
            else:
                idx-=len(d)
        
        return None

vctk_audio=VCTKAudio(vctk, portion=(0, int(len(vctk)*0.8)))
vctk_test=VCTKAudio(vctk, portion=(int(len(vctk)*0.8), len(vctk)))

wham=WHAM('./wham_noise/tr')
wham_test=WHAM('./wham_noise/cv')

# MS dataset
class MS_SNSD:
    def __init__(self, path, shuffle=True, test=True, train=True):
        noisy_wav_dir_path = []
        if test:
            noisy_wav_dir_path.append(path+'/noise_test')
        if train:
            noisy_wav_dir_path.append(path+'/noise_train')
            
        self.noisy_files = []
        for noisy_wav_dir in noisy_wav_dir_path:
            for file in os.listdir(noisy_wav_dir):
                if file.endswith('.wav'):
                    self.noisy_files.append(os.path.join(noisy_wav_dir, file))
        if shuffle:
            np.random.shuffle(self.noisy_files)
    def __len__(self):
        return len(self.noisy_files)
    def __read_audio(self,file):
        noise_audio, _ = librosa.load(file, sr=fs, mono=True)
        return noise_audio
    def __getitem__(self, idx):
        return self.__read_audio(self.noisy_files[idx])
    def get_batch(self, idx1, idx2):
        mic_sig_batch = []
        for idx in range(idx1, idx2):
            mic_sig_batch.append(self.__read_audio(self.noisy_files[idx]))
            
        return np.stack(mic_sig_batch)

ms_snsd=MS_SNSD('./MS-SNSD', test=False, train=True)
ms_snsd_test=MS_SNSD('./MS-SNSD', test=True, train=False)

# define datasets
noisesets=FuseDataset(ms_snsd, wham)
noisesets_test=FuseDataset(ms_snsd_test, wham_test)
audiosets=vctk_audio
audiosets_test=vctk_test

# acoustic property
N_MIC=6
R_MIC=0.0463

import pyroomacoustics as pra
def generate_mic_array(mic_radius: float, n_mics: int, pos):
    """
    Generate a list of Microphone objects
    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[pos[0], pos[1]], M=n_mics, phi0=0, radius=mic_radius)
    R=np.concatenate((R, np.ones((1, n_mics))*pos[2]), axis=0)
    return R

# global mic array:
R_global=generate_mic_array(R_MIC, N_MIC, (0,0,0))

# simulate the room

def get_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_angle(px, py):
    return np.arctan2(py, px)

def random2D(range_x, range_y, except_x, except_y, except_r, retry=100):
    if retry==0: return None
    if except_r<except_x<range_x-except_r and except_r<except_y<range_y-except_r:
        loc=(np.random.uniform(0, range_x), np.random.uniform(0, range_y))
        if get_dist(loc, (except_x, except_y))<except_r:
            return random2D(range_x, range_y, except_x, except_y, except_r, retry-1)
        else:
            return loc
    else:
        return None
        
def simulateRoom(N_source, min_room_dim, max_room_dim, min_gap, min_dist, min_angle_diff=0, retry=20):
    if retry==0: 
        print("warning: room simulation failed")
        return None
    # return simulated room
    room_dim=[np.random.uniform(low, high) for low, high in zip(min_room_dim, max_room_dim)]
    R_loc=[np.random.uniform(min_gap, x-min_gap) for x in room_dim]
    source_locations=[random2D(room_dim[0], room_dim[1], R_loc[0], R_loc[1], min_dist) for i in range(N_source)]
    if None in source_locations: return None
    
    angles=[get_angle(p[0]-R_loc[0], p[1]-R_loc[1]) for p in source_locations]
    if N_source>1:
        min_angle_diff_rad=min_angle_diff*np.pi/180
        angles_sorted=np.sort(angles)
        if np.min(angles_sorted[1:]-angles_sorted[:-1])<min_angle_diff_rad or angles_sorted[0]-angles_sorted[-1]+2*np.pi<min_angle_diff_rad:
            return simulateRoom(N_source, min_room_dim, max_room_dim, min_gap, min_dist, min_angle_diff, retry-1)
    
    source_locations=[(x,y,R_loc[2]) for x,y in source_locations]
    
    return (room_dim, R_loc, source_locations, angles)

# Define materials
wall_materials = [
    'curtains_velvet',
    'felt_5mm',
    'rough_concrete',
    'plasterboard',
    'wood_1.6cm',
    'smooth_brickwork_flush_pointing',
    'smooth_brickwork_10mm_pointing',
    'brick_wall_rough',
    'fibre_absorber_1',
    'hanging_absorber_panels_1',
    'limestone_wall'
]
floor_materials = [
    'linoleum_on_concrete',
    'carpet_cotton',
    'carpet_tufted_9.5mm',
    'carpet_thin',
    'carpet_6mm_closed_cell_foam',
    'carpet_6mm_open_cell_foam',
    'carpet_tufted_9m',
    'felt_5mm',
    'carpet_soft_10mm',
    'carpet_hairy',
]

def simulateSound(room_dim, R_loc, source_locations, source_audios, rt60, materials=None, max_order=None):
    # source_audios: array of numpy array
    # L: max of all audios. Zero padding at the end
    # return (all_channel_data (C, L), groundtruth_with_reverb (N, C, L), groundtruth_data (N, C, L), angles (N)
    
    if materials is not None:
        (ceiling, east, west, north, south, floor)=materials
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.make_materials(
                ceiling=ceiling,
                floor=floor,
                east=east,
                west=west,
                north=north,
                south=south,
            ), max_order=max_order
        )
    else:
        try:
            e_absorption, max_order_rt60 = pra.inverse_sabine(rt60, room_dim)    
        except ValueError:
            e_absorption, max_order_rt60 = pra.inverse_sabine(1, room_dim)
        room=pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=min(max_order_rt60, max_order))
    
    R=generate_mic_array(R_MIC, N_MIC, R_loc)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    length=max([len(source_audios[i]) for i in range(len(source_audios))])
    for i in range(len(source_audios)):
        source_audios[i]=np.pad(source_audios[i], (0, length-len(source_audios[i])), 'constant')
        
    
    for i in range(len(source_locations)):
        room.add_source(source_locations[i], signal=source_audios[i], delay=0)
    
    room.image_source_model()
    premix_w_reverb=room.simulate(return_premix=True)
    mixed=room.mic_array.signals
    
    # groundtruth
    room_gt=pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(1.0), max_order=0)
    
    new_angles=np.zeros((len(source_locations),))
    
    R_gt=np.mean(R, axis=-1).reshape((3,1))
    room_gt.add_microphone_array(pra.MicrophoneArray(R_gt, room.fs))
    
    for i in range(len(source_locations)):
        room_gt.add_source(source_locations[i], signal=source_audios[i], delay=0)
        new_angles[i]=get_angle(source_locations[i][0]-R_gt[0,0], source_locations[i][1]-R_gt[1,0])
    room_gt.compute_rir()
    
    room_gt.image_source_model()
    premix=room_gt.simulate(return_premix=True)
    
    return (mixed, premix_w_reverb, premix, new_angles)

def simulateBackground(background_audio):
    # diffused noise. simulate in a large room
    bg_radius = np.random.uniform(low=10.0, high=20.0)
    bg_theta = np.random.uniform(low=0, high=2 * np.pi)
    H=10
    bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta), H]

    # Bg should be further away to be diffuse
    left_wall = np.random.uniform(low=-40, high=-20)
    right_wall = np.random.uniform(low=20, high=40)
    top_wall = np.random.uniform(low=20, high=40)
    bottom_wall = np.random.uniform(low=-40, high=-20)
    height = np.random.uniform(low=20, high=40)
    corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                    [   right_wall, top_wall], [right_wall, bottom_wall]]).T
    absorption = np.random.uniform(low=0.1, high=0.5)
    room = pra.Room.from_corners(corners,
                                 fs=fs,
                                 max_order=10,
                                 materials=pra.Material(absorption))
    room.extrude(height)
    mic_array = generate_mic_array(R_MIC, N_MIC, (0,0,H))
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs))
    room.add_source(bg_loc, signal=background_audio)

    room.image_source_model()
    room.simulate()
    return room.mic_array.signals

# Audio simulation

from util import power, mix
from torch.utils.data import Dataset


class OnlineSimulationDataset(Dataset):
    def __init__(self, voice_collection, noise_collection, length, simulation_config, truncator, cache_folder, cache_max=None):
        self.voices=voice_collection
        self.noises=noise_collection
        self.length=length
        self.seed=simulation_config['seed']
        self.additive_noise_min_snr=simulation_config['min_snr']
        self.additive_noise_max_snr=simulation_config['max_snr']
        self.special_noise_ratio=simulation_config['special_noise_ratio']
        self.source_dist=simulation_config['source_dist']
        self.min_angle_diff=simulation_config['min_angle_diff']
        self.max_rt60=simulation_config['max_rt60'] 
        self.min_rt60=0.15 # minimum to satisfy room odometry
        self.max_room_dim=simulation_config['max_room_dim'] 
        self.min_room_dim=simulation_config['min_room_dim'] 
        self.min_dist=simulation_config['min_dist'] 
        self.min_gap=simulation_config['min_gap'] 
        self.max_order=simulation_config['max_order']
        self.randomize_material_ratio=simulation_config['randomize_material_ratio']
        self.max_latency=simulation_config['max_latency']
        self.random_volume_range=simulation_config['random_volume_range'] 
        
        self.low_volume_ratio=simulation_config['low_volume_ratio']
        self.low_volume_range=simulation_config['low_volume_range']
        self.angle_dev=simulation_config['angle_dev']
        
        self.no_reverb_ratio=simulation_config['no_reverb_ratio']
        
        self.truncator=truncator
        self.cache_folder=cache_folder
        self.cache_history=[]
        self.cache_max=cache_max
        
    def __seed_for_idx(self,idx):
        return self.seed+idx
    
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # return format: 
        # (
        # mixed multichannel audio, (C,L)
        # array of groundtruth with reverb for each target, (N, C, L)
        # array of direction of targets, (N,)
        # array of multichannel ideal groundtruths for each target, (N, C, L)
        # noise (C, L)
        # )
        # check cache first
        
        if idx>=self.length:
            return None
        
        if self.cache_folder is not None:
            cache_path=self.cache_folder+'/'+str(idx)+'-'+str(self.seed)+'.npz'
            
            if cache_path not in self.cache_history:
                self.cache_history.append(cache_path)
            
                if self.cache_max is not None and self.cache_max==len(self.cache_history):
                    # delete first one
                    first=self.cache_history[0]
                    os.remove(first)
                    self.cache_history=self.cache_history[1:]
                
            if os.path.exists(cache_path):
                cache_result=np.load(cache_path, allow_pickle=True)['data']
                return cache_result[0], cache_result[1], cache_result[2], cache_result[3], cache_result[4]
        else:
            cache_path=None
        
        
        np.random.seed(self.__seed_for_idx(idx))
        n_source=np.random.choice(np.arange(len(self.source_dist)), p=self.source_dist)+1
        
        room_result=simulateRoom(n_source, self.min_room_dim, self.max_room_dim, self.min_gap, self.min_dist, self.min_angle_diff)
        if room_result is None:
            return self.__getitem__(idx+1) # backoff
            
        room_dim, R_loc, source_loc, source_angles=room_result
        
        voices=[self.truncator.process(self.voices[vi]) for vi in np.random.choice(len(self.voices), n_source)]
        
        # normalize voice
        voices=[v/np.max(np.abs(v)) for v in voices]
        
        voices=[v*np.random.uniform(self.random_volume_range[0], self.random_volume_range[1]) for v in voices]
        if np.random.rand()<self.low_volume_ratio:
            voices[0]*=np.random.uniform(self.low_volume_range[0], self.low_volume_range[1])
        
        if self.special_noise_ratio>np.random.rand():
            noise=self.truncator.process(self.noises[np.random.choice(len(self.noises))])
        else:
            noise=np.random.randn(self.truncator.get_length())

        no_reverb=(np.random.rand()<self.no_reverb_ratio)
        max_order=0 if no_reverb else self.max_order
        if self.randomize_material_ratio>np.random.rand():
            ceiling, east, west, north, south = tuple(np.random.choice(wall_materials, 5))  # sample material
            floor = np.random.choice(floor_materials)  # sample material
            ceiling=np.random.choice(floor_materials)
            mixed, premix_w_reverb, premix, new_angles=simulateSound(room_dim, R_loc, source_loc, voices, 0, (ceiling, east, west, north, south, floor), max_order)
        else:
            rt60=np.random.uniform(self.min_rt60, self.max_rt60)
            mixed, premix_w_reverb, premix, new_angles=simulateSound(room_dim, R_loc, source_loc, voices, rt60, None, max_order)
        
        
        background=simulateBackground(noise)
        snr=np.random.uniform(self.additive_noise_min_snr, self.additive_noise_max_snr)
        
        # trucate to the same length
        mixed=mixed[:, :truncator.get_length()]
        background=background[:, :truncator.get_length()]
        
        total, background=mix(mixed, background, snr)
        
        new_angles[0]+=(np.random.rand()*2-1)*self.angle_dev

        
        # save cache
        if cache_path is not None:
            np.savez_compressed(cache_path, data=[total, premix_w_reverb, new_angles, premix, background])
        
        return total, premix_w_reverb, new_angles, premix, background

# randomly truncate audio to fixed length
class RandomTruncate:
    def __init__(self, target_length, seed, power_threshold=None):
        self.length=target_length
        self.seed=seed
        self.power_threshold=power_threshold
        np.random.seed(seed)
    
    def process(self, audio):
        # if there is a threshold
        if self.power_threshold is not None:
            # smooth
            power=np.convolve(audio**2, np.ones((32,)), 'same')
            avgpower=np.mean(power)
            for i in range(len(power)):
                # threshold*mean_power
                if power[i]>avgpower*self.power_threshold:
                    # leave ~=0.3s of start
                    fs=48000
                    audio=audio[max(0, i-int(0.3*fs)):]
                    break
        if len(audio)<self.length:
            nfront=np.random.randint(self.length-len(audio))
            return np.pad(audio, (nfront, self.length-len(audio)-nfront), 'constant')
        elif len(audio)==self.length:
            return audio
        else:
            start=np.random.randint(len(audio)-self.length)
            return audio[start:start+self.length]
        
    def get_length(self):
        return self.length
        

# test simulation dataset and truncate
simulation_config={
    'seed':3,
    'min_snr':5,
    'max_snr':25,
    'special_noise_ratio':0.9,
    'source_dist':[0.1, 0.4, 0.4, 0.1],
    'min_angle_diff':20,
    'max_rt60': 0.5,
    'max_room_dim':[7,7,4],
    'min_room_dim':[3,3,2],
    'min_dist': 0.8,
    'min_gap': 1.2,
    'max_order':8,
    'randomize_material_ratio':0.8,
    'max_latency':0.3,
    'random_volume_range': [0.7, 1],
    'low_volume_ratio': 0.04,
    'low_volume_range': [0.01, 0.02],
    'angle_dev': 0.09, # ~5 degree
    'no_reverb_ratio':0
}
truncator=RandomTruncate(4*fs, 5, 0.4)

# no cache for the original dataset
dataset=OnlineSimulationDataset(audiosets, noisesets, 8000, simulation_config, truncator, None)

# beamformer test
import beamformer
import importlib
importlib.reload(beamformer)
from beamformer import WebrtcBeamformer, OnlineMVDRBeamformer
buffer_size_webrtc=384 # 8ms
buffer_size_mvdr=384
webrtc_path='LD_LIBRARY_PATH=./lib/ ./beamform_mic_array'

beamformers=[WebrtcBeamformer(buffer_size_webrtc, R_global, fs, webrtc_path, cache_path), 
             OnlineMVDRBeamformer(buffer_size_mvdr, R_global, fs, V, False), # superdirective
             OnlineMVDRBeamformer(buffer_size_mvdr, R_global, fs, V, True, True, False) # online mvdr
            ]

from util import alignChannel
from scipy.signal import resample

# resample to 16kHz
def batchOneThirdSR(signals):
    if len(signals.shape)==2:
        n=signals.shape[0]
        result=np.zeros((n, signals.shape[1]//3))
        for i in range(n):
            result[i]=resample(signals[i], signals.shape[1]//3)
        return result
    else:
        result=np.stack([batchOneThirdSR(signals[i]) for i in range(signals.shape[0])], axis=0)
        
class SimulationBeamformedDataset():
    def __init__(self, simulationDataset, beamformers, cache_folder, cache_max=None, align=False, sample_offset=0):
        self.beamformers=beamformers
        self.dataset=simulationDataset
        
        self.cache_folder=cache_folder
        self.cache_history=[]
        self.cache_max=cache_max
        self.align=align
        
        self.sample_offset=sample_offset
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        
        if self.cache_folder is not None:
            cache_path=self.cache_folder+'/sbd-'+str(idx)+'.npz'
            
            if cache_path not in self.cache_history:
                self.cache_history.append(cache_path)
            
                if self.cache_max is not None and self.cache_max==len(self.cache_history):
                    # delete first one
                    first=self.cache_history[0]
                    os.remove(first)
                    self.cache_history=self.cache_history[1:]
                
            if os.path.exists(cache_path):
                cache_result=np.load(cache_path, allow_pickle=True)['data']
                cache_result[0]=cache_result[0][..., self.sample_offset:]
                cache_result[1]=cache_result[1][..., self.sample_offset:]
                cache_result[2]=cache_result[2][..., self.sample_offset:]
                return cache_result[0], cache_result[1], cache_result[2], cache_result[3]
        else:
            cache_path=None
        
        total, premix_w_reverb, source_angles, premix, background=self.dataset[idx]
        # choose the first component
        bfdata=np.zeros((len(self.beamformers), total.shape[1]))
        for i in range(len(self.beamformers)):
            res=self.beamformers[i].process(total.T, source_angles[0])[:total.shape[1]]
            if len(res)<total.shape[1]: 
                bfdata[i]=np.pad(res, (0, total.shape[1]-len(res)), 'constant')
            else:
                bfdata[i]=res[:total.shape[1]]
        if self.align:
            total=alignChannel(total.T, source_angles[0], R_global-np.tile(np.mean(R_global, axis=-1).reshape((3,1)), (1, N_MIC))).T
                
        # currently only need total, bfdata and premix
        gt=premix[0, :, :total.shape[1]]
        
        total=batchOneThirdSR(total)
        bfdata=batchOneThirdSR(bfdata)
        gt=batchOneThirdSR(gt)
        
        # save cache
        if cache_path is not None:
            np.savez_compressed(cache_path, data=[total, bfdata, gt, source_angles[0]])
        
        total=total[..., self.sample_offset:]
        bfdata=bfdata[..., self.sample_offset:]
        gt=gt[..., self.sample_offset:]        
        
        return total, bfdata, gt, source_angles[0]


import util
from util import si_sdr, calculate_gain

train_dataset=SimulationBeamformedDataset(dataset, beamformers, './bf_cache', 8100, True, sample_offset=8)

simulation_config_test={
    'seed':5,
    'min_snr':5,
    'max_snr':25,
    'special_noise_ratio':0.9,
    'source_dist':[0.1, 0.4, 0.4, 0.1],
    'min_angle_diff':20,
    'max_rt60': 0.5,
    'max_room_dim':[7,7,4],
    'min_room_dim':[3,3,2],
    'min_dist': 0.8,
    'min_gap': 1.2,
    'max_order':8,
    'randomize_material_ratio':0.8,
    'max_latency':0.3,
    'random_volume_range': [0.7, 1],
    'low_volume_ratio': 0.04,
    'low_volume_range': [0.01, 0.02],
    'angle_dev': 0.09, # ~5 degree
    'no_reverb_ratio':0
}

testset=OnlineSimulationDataset(audiosets_test,noisesets_test, 300, simulation_config_test, truncator, None)
test_dataset=SimulationBeamformedDataset(testset, beamformers, './bf_cache_test', 600, True, sample_offset=8)

import time
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

BATCH=40
batch_mul=1

cuda_id=None

dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
testloader=torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4)

def train_epoch(model, lossmodel, optimizer, scheduler, dataloader, save_path=None, last_loss=None):
    model.train()
    losses=[]
    tick0=time.time()
    high_cnt=0
    low_cnt=0
    
    optimizer.zero_grad()
    
    batch_losses=[]
    nan=False
    for batch_idx, (total, bfdata, premix, angles) in enumerate(dataloader):
        tick1=time.time()
        data_all=torch.Tensor(np.concatenate([total, bfdata], axis=1))
        data=data_all.cuda(cuda_id)
        premix_gpu=premix.cuda(cuda_id)
        tick2=time.time()
        output=model(data)
        loss=lossmodel(output, premix_gpu).sum()
        l=loss.item()
        tick3=time.time()
        batch_losses.append(l)
        if batch_idx%batch_mul==batch_mul-1:
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            
            # check nan
            if not np.isnan(batch_losses).any():
                optimizer.step()
                print(sum(batch_losses))
            else:
                print("NaN loss, skip")
                nan=True
                break
            
            optimizer.zero_grad()
            batch_losses=[]
            
        
        data=None
        premix=None
        losses.append(l)
        tick0=tick3
        
    scheduler.step()
    print("scheduler lr: ", scheduler.get_last_lr()[0])
    print("selective back: ", low_cnt, high_cnt)
    
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print("saved")
    
    if not nan:
        return np.mean(losses)
    else:
        return None

from deepbeam import BeamformerModel

class FuseLoss(nn.Module):
    def __init__(self, offset, r=50, lookahead=0):
        super().__init__()
        self.offset=offset
        self.l1loss=nn.L1Loss()
        self.sisdrloss=asteroid.losses.pairwise_neg_sisdr
        self.r=r
        self.lookahead=lookahead
        
    def forward(self, signal, gt):
        if len(signal.shape)==2:
            signal=signal.unsqueeze(1)
        if self.lookahead==0:
            return self.l1loss(signal[..., self.offset:], gt[..., self.offset:])*self.r+torch.mean(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
        else:
            return self.l1loss(signal[..., self.offset+self.lookahead:], gt[..., self.offset:-self.lookahead])*self.r+torch.mean(self.sisdrloss(signal[..., self.offset+self.lookahead:], gt[..., self.offset:-self.lookahead]))
    

network=BeamformerModel(ch_in=9, synth_mid=64, synth_hid=96, block_size=16, kernel=3, synth_layer=4, synth_rep=4, lookahead=0)
loss_net=FuseLoss(12000, 10)

network=network.cuda(cuda_id)
loss_net=loss_net.cuda(cuda_id)

# evaluate
def test_epoch(model, lossmodel, testloader):
    model.eval()
    gains=[]
    losses=[]
    inputgains=[]
    with torch.no_grad():
        for batch_idx, (total, bfdata, premix, angles) in enumerate(testloader):
            data_all=torch.Tensor(np.concatenate([total, bfdata], axis=1))
            data=data_all.cuda(cuda_id)
            premix_gpu=premix.cuda(cuda_id)
            
            output=model(data)
            
            loss=lossmodel(output, premix_gpu).sum()
            losses.append(loss.item())
            # test si-sdr
            output_cpu=output.cpu().detach().numpy()
            for i in range(output_cpu.shape[0]):
                if torch.max(premix[i,0])>0.02:
                    gain, _,_=calculate_gain(premix[i, 0, 12000:], total[i, 0,12000:], output_cpu[i, 0,12000:])
                    gains.append(gain)
                    gain, _,_=calculate_gain(premix[i,0], total[i,0], bfdata[i,1])
                    inputgains.append(gain)
    
    return losses, gains, inputgains


def train():
    train_losses=[]
    test_losses=[]

    opt1=optim.AdamW(network.parameters(), lr=3e-4)
    opt2=optim.AdamW(network.parameters(), lr=1e-4)

    opts=[(opt1, StepLR(opt1, step_size=25, gamma=0.7), 25), # 2
        (opt2, StepLR(opt2, step_size=150, gamma=0.6), 150)]

    for (opt, sch, n) in opts:
        for i in range(n):
            print("epoch "+str(len(train_losses)))
            last_loss=None
            res=train_epoch(network, loss_net, opt, sch, dataloader, './trained/deepbeam_p6.bin', last_loss)
            losses, gains, inputgains=test_epoch(network, loss_net, testloader)
            test_losses.append(np.mean(losses))
            print("avg: ", np.mean(losses), np.mean(gains), np.mean(inputgains))
            if res is not None:
                train_losses.append(res)
            else:
                break

            print("loss", res)

def generate_data():
    print("Generating training data...")
    for batch_idx, (total, bfdata, premix, angles) in enumerate(dataloader):
        print(batch_idx,'/', len(dataloader))
        pass
    print("Generating test data...")
    for batch_idx, (total, bfdata, premix, angles) in enumerate(testloader):
        print(batch_idx, '/', len(testloader))
        pass


def load_path(model, path):
    model.load_state_dict(torch.load(path), strict=False)

def test_data(modelpath, filepath, destination):
    load_path(network, modelpath)
    sig=librosa.load(filepath, sr=16000, mono=False)[0]
    result=network(sig[..., :-8])
    from scipy.io import wavfile
    wavfile.write(destination, 16000, result.reshape((-1,)))

import sys
if __name__ == "__main__":
    if sys.argv[1]=='data':
        generate_data()
    elif sys.argv[1]=='train':
        train()
    elif sys.argv[1]=='test':
        modelpath=sys.argv[2]
        filepath=sys.argv[3]
        dest=sys.argv[4]
        test_data(modelpath, filepath, dest)
