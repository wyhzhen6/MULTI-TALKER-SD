import argparse
import os
import subprocess
import re
import soundfile as sf
import json
import numpy as np
import csv
import pandas as pd


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pyroomacoustics as pra
import time

import math
import random
from typing import List, Tuple
from numpy.linalg import norm
import torch

from scipy.signal import convolve
from collections import defaultdict
from scipy.signal import fftconvolve

class singlechannel_rir_room:
    def __init__(self, 
                filepath,
                *,
                simulate_config: str,
                speech_host_label=None,
                
        ):
        self.filepath = filepath
        self.simulate_config = simulate_config
        
        self.meeting_type = simulate_config['meeting_type']
        self.room_size = simulate_config['room_size']
        self.room_size_mid = simulate_config['room_size_mid']
        self.room_size_lar = simulate_config['room_size_lar']
        self.room_type = simulate_config['room_type']
        self.rt60 = simulate_config['rt60']
        self.rt60_mid = simulate_config['rt60_mid']
        self.rt60_lar = simulate_config['rt60_lar']
        self.fs = simulate_config.get('fs', 16000)
        self.is_compute_DRR = simulate_config['is_compute_DRR']
        self.is_compute_SRR = simulate_config['is_compute_SRR']
        self.mic_loc = simulate_config['mic_pos']
        self.signal_gains_arr = simulate_config['signal_gains']
        
        self.speech_host_label = speech_host_label

        # collect metadata
        self.audio_len = 0.0    # result audio length
        self.listdata = []      # metadata: list[ dict {"start_time", "end_time", "speaker", "path"} ]
        self.vad_dur = 0        # speech duration
        self._read_listfile2(filepath)
        
        # setting room parameters and create
        self._generate_room_pra(["desk", "circle", "speech"])
        self.room = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            materials=pra.Material(self.e_absorption),
            max_order=self.max_order,
            ray_tracing=True,
            air_absorption=True,
        )

        # setting distance parameters
        self.d = 0.3        # min_dis
        self.d_src = round(random.uniform(0.4, 0.5), 2) # min_src_dis
        self.d_wall = round(random.uniform(0.45, 0.55), 2) # min_src_wall

        # setting rir parameters
        self._generate_src_pra()
        self._create_mic()
        
        # prepare noise type, path
        self._prepare_point_noise()
        
        # others
        self.max_attempts = 10000   # max_attempts for set speaker pos
        self.angles = []            # src_angles circle for `circle``
        self.loc = []               # src_loc
        self.DRR = []
        self.SRR = []
        self.host_audio = []        # audio for merging host_audio, computing host_SRR
        self.gains = []             # gains for speakers
        
        self.host_pos = []
        self.host_delay = []
        self.drr = 0
        self.srr = 0
        
        # nosie
        self.SNR_point = 0
        self.SNR_diffuse = 0
        self.point_noise_time = 0
        self.avg_snr = []
        
    def to_dic(self):
        return {
            "meeting_type" : self.meeting_type,
            "room_size" : self.room_size,
            "room_type" :self.room_type,
            "rt60 " :self.rt60,
            "fs " :self.fs,
            "host_label" :self.speech_host_label,
            "src_num " :len(self.listdata),
            "src_pos " : self.loc,
            "SRR " :self.srr,
            "DRR " :self.drr,
            "point noise SNR" :self.SNR_point,
            "diffuse noise SNR " :self.SNR_diffuse,
            "gain": self.gains,
            "avg_SNR" : self.avg_snr,
            "audio_len " :self.audio_len,
            "vad_dur " :self.vad_dur
        }
    def to_json(self,indent=4):
        return json.dumps(self.to_dic(), ensure_ascii=False, indent=indent)
    
    
    def _prepare_point_noise(self):
        '''
            category_files: { "category": [file1, file2] }
            target_categories: { "category": "postion" }
        '''
        noise_csv_path = self.simulate_config['point_noise_csv']
        typelist_path = self.simulate_config['point_noise_type']
        
        self.target_categories = []
        with open(typelist_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    category = parts[0]
                    position = parts[1]
                    self.target_categories.append((category, position))

        df = pd.read_csv(noise_csv_path)

        self.category_files = defaultdict(list)
        for idx, row in df.iterrows():
            filename = row['filename']
            category = row['category']
            self.category_files[category].append(filename)

    
    def _read_listfile2(self, filename):
        '''
            load metadata
            calculate speech duration
            listdata: list[ dict {"start_time", "end_time", "label", "speaker", "path"} ]
        '''
        with open(filename, "r") as f:
            lines = [line.strip() for line in f]
        
        vad_label = []
        for line in lines:
            parts = line.split()
            start_time = float(parts[0])
            end_time = float(parts[1])
            if len(parts) >= 4:
                vad_label.append((start_time, end_time)) # 
                index = next((item for item in self.listdata if item["label"] == parts[2]), None)
                if index is not None:
                    if end_time > index["end"]:
                        index["end"] = end_time
                    index["start_time"].append(start_time)
                    index["end_time"].append(end_time)
                    index["file_path"].append(parts[3])
                else:
                    self.listdata.append({
                        "start_time": [start_time],  
                        "end_time": [end_time],    
                        "label": parts[2],              
                        "file_path": [parts[3]],
                        "end": float(parts[1]),
                    })

                if end_time > self.audio_len:
                    self.audio_len = end_time

        # calc speech duration
        sorted_segments = sorted(vad_label, key=lambda x: x[0])
        merged = []
        current_start, current_end = sorted_segments[0]
        for start, end in sorted_segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        self.vad_dur = sum(end - start for start, end in merged)

    
    def _generate_room_pra(self, meeting_types):
        '''
            generate parameters for rir
            rt60
            room_size
            e_absorption
            max_order
            meeting_type
        '''
        if self.rt60 == None:
            if len(self.listdata)<20:
                self.rt60 = round(random.uniform(self.rt60_mid[0], self.rt60_mid[1]), 2)
            else:
                self.rt60 = round(random.uniform(self.rt60_lar[0], self.rt60_lar[1]), 2)
                
        if self.room_size == None:
            if len(self.listdata) < 20:
                size_min = self.room_size_mid[0]
                size_max = self.room_size_mid[1]
                self.room_size = [random.uniform(size_min[0],size_max[0]), random.uniform(size_min[1],size_max[1]), random.uniform(size_min[2],size_max[2])]
                self.room_type = "middle"
            else:
                size_min = self.room_size_lar[0]
                size_max = self.room_size_lar[1]
                self.room_size = [random.uniform(size_min[0],size_max[0]), random.uniform(size_min[1],size_max[1]), random.uniform(size_min[2],size_max[2])]
                self.room_type = "large"
 
        if self.meeting_type == None:
            if 'pre' in self.filepath:
                self.meeting_type = 'speech'
            else:
                meeting_types = [meeting_types[0], meeting_types[2]]
                self.meeting_type = random.choice(meeting_types)
                
        #compute e_absorption, max_order
        self.e_absorption, self.max_order = pra.inverse_sabine(self.rt60,self.room_size)

    def _generate_src_pra(self):
        '''
            generate parameters for different meeting_type
            "circle":
                src_center: middle of the room
                max_radius: half of short side - min_distance to wall(0.5-0.7)
                radius: randomly from 2 to max_radius
                min_angle
            "desk":
                width: room_size[0]-2*self.d_wall
                length: room_size[1]-self.d_wall
            "speech":
                src_center: middle of the audience area(room - host area)
                max_radius: max radius for audience_pos

        '''
        if self.meeting_type == "circle":
            self.src_center = [(self.room_size[0]-self.d_wall)/2,self.room_size[1]/2]
            self.max_radius = min(self.room_size[0]-self.d_wall,self.room_size[1])/2-self.d_wall
            self.radius = round(random.uniform(2,self.max_radius),3)
            # Modified: self.min_angle is now set to half of the original value to reduce the minimum angle between two speakers.
            # Original: self.min_angle = 2 * math.asin(self.d_wall / (2 * self.radius))
            # Changed to half of the original value
            self.min_angle = math.asin(self.d_wall / (2 * self.radius))
        elif self.meeting_type == "desk":
            self.width = self.room_size[0]-2*self.d_wall
            self.length = self.room_size[1]-self.d_wall
        elif self.meeting_type == "speech":
            self.src_center = [(self.room_size[0]-4*self.d_wall)/2,self.room_size[1]/2]
            self.max_radius = min(self.src_center[0],self.src_center[1])-self.d_wall

    def _create_mic(self):
        '''
            create microphone
            mic_pos: middle of the room
            mic_height: 
                "circle": table(0.8 to 1)
                "desk": table(0.8 to 1)
                "speech": ceiling(room_height-1 to room_height-min_mic_dis)
        '''

        mic_height =[round(random.uniform(0.8,1),2),round(random.uniform(self.room_size[2]-1,self.room_size[2]-self.d),2)] #table,ceiling
        
        if self.mic_loc == None:
            if self.meeting_type == "circle":
                h = mic_height[0] #table
            elif self.meeting_type == "desk":
                h = mic_height[0] #table
            elif self.meeting_type == "speech":
                h = mic_height[1] #ceiling

            self.mic_loc = [self.room_size[0]/2, self.room_size[1]/2, h]
        self.room.add_microphone(self.mic_loc)

    def _set_host_label(self, host_label):
        '''
            set host for speech type
            default: 1st label of listdata
        '''
        if host_label is not None:
            index = next((item for item in self.listdata if item["label"] == host_label), None)
            if index is not None: 
                self.speech_host_label = host_label
                return 
        
        item = self.listdata[0]
        host_label = item["label"]
        self.speech_host_label = host_label
    
    def simulate(self, output_dir):
        '''
        rir simulate
        '''
        # Set host label for 'speech' meeting type
        if self.meeting_type == "speech":
            host_label = self.simulate_config.get('host_label', )
            self._set_host_label(host_label)

        total_length = int(self.audio_len * self.fs) + \
            16000  # add margin for tail
        
        # Initialize output signals for each array
        signals = np.zeros((1, total_length), dtype=np.float32)

        # Generate RIRs for each source and each array
        rir_dict = []
        src_positions = []
        host_indices = []
        for idx, item in enumerate(self.listdata):
            if item["label"] != self.speech_host_label:
                src_pos = self._set_pos()
                src_positions.append(src_pos)
                self.room.add_source(src_pos)
        # Place host position at the end
        if self.meeting_type == "speech":
            item = next(
                (item for item in self.listdata if item["label"] == self.speech_host_label), None)
            self._set_speech_host_pos(item)
            for hpos in self.host_pos:
                src_positions.append(hpos)
                self.room.add_source(hpos)
            # host_indices
            host_indices = list(
                range(len(src_positions) - len(self.host_pos), len(src_positions)))

        self.room.compute_rir()

        # RIR Extraction
        for src_idx, src_pos in enumerate(src_positions):
            rir = self.room.rir[0][src_idx]
            rir_dict.append(rir)
        
        # Segments convolution for all speakers
        for src_idx, item in enumerate(self.listdata):
            signal_gain = random.uniform(
                self.signal_gains_arr[0], self.signal_gains_arr[1])
            self.gains.append(signal_gain)
            if item["label"] == self.speech_host_label:
                # host part:
                for host_idx, audio_host in enumerate(self.host_audio):
                    # start_sample = int(item["start_time"][host_idx] * self.fs)
                    start_sample = int(self.host_delay[host_idx] * self.fs)
                    host_src_idx = host_indices[host_idx]
                    rir = rir_dict[host_src_idx]
                    conv = fftconvolve(audio_host, rir)
                    end_idx = min(
                        start_sample + len(conv), total_length)
                    signals[0][start_sample:end_idx] += conv[:end_idx - start_sample]
            else:
                # non-host speakers
                for seg_idx in range(len(item["file_path"])):
                    audio_path = item["file_path"][seg_idx]
                    audio, fs = sf.read(audio_path)
                    if fs != self.fs:
                        audio = self.resample(audio, fs, self.fs)
                    if len(audio.shape) > 1:
                        audio = audio[0]
                    st_time = int(item["start_time"][seg_idx] * self.fs)
                    ed_time = int(item["end_time"][seg_idx] * self.fs)
                    seg_audio = audio[:ed_time - st_time]
                    seg_audio = self._set_gain(
                        torch.from_numpy(seg_audio), signal_gain).numpy()

                    rir = rir_dict[src_idx]
                    conv = fftconvolve(seg_audio, rir)
                    end_idx = min(st_time + len(conv), total_length)
                    signals[0][st_time:end_idx] += conv[:end_idx - st_time]

        self.room.mic_array.record(signals, self.fs)
        
        # SRR/DRR calculation (new version, directly use rir_dict and host_audio/seg_audio)
        if self.is_compute_SRR or self.is_compute_DRR:
            # Non-host speakers
            idx = 0
            for src_idx, item in enumerate(self.listdata):
                if item["label"] == self.speech_host_label:
                    continue
                # Only use the first segment audio (extend if needed)
                audio_path = item["file_path"][0]
                audio, fs = sf.read(audio_path)
                if fs != self.fs:
                    audio = self.resample(audio, fs, self.fs)
                if len(audio.shape) > 1:
                    audio = audio[0]
                seg_audio = audio
                # Use the first channel of the first mic array RIR
                rir = rir_dict[idx]
                self.SRR.append(
                    self._compute_SRR(seg_audio, rir, self.fs))
                self.DRR.append(
                    self._compute_DRR(rir, self.fs))

                idx += 1

            # Host speaker
            if self.meeting_type == "speech":
                item = next(
                    (item for item in self.listdata if item["label"] == self.speech_host_label), None)
                host_srr = []
                host_drr = []
                for host_idx, audio_host in enumerate(self.host_audio):
                    rir = rir_dict[host_indices[host_idx]]
                    host_srr.append(
                        self._compute_SRR(audio_host, rir, self.fs))
                    host_drr.append(
                        self._compute_DRR(rir, self.fs))
                
                self.SRR.append(np.mean(host_srr))
                self.DRR.append(np.mean(host_drr))

        if self.is_compute_SRR:
            self.srr = np.mean(self.SRR)

        if self.is_compute_DRR:
            self.drr = np.mean(self.DRR)

        #save reverb audio
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        reverb_signal = np.asarray(self.room.mic_array.signals, dtype=np.float32)
        reverb_signal = reverb_signal.T
        output_path_reverb = os.path.join(output_dir,"reverb", f"{filename}_reverb.wav")
        os.makedirs(os.path.dirname(output_path_reverb), exist_ok=True)
        sf.write(
            output_path_reverb,  
            reverb_signal,         
            self.fs,                   
            subtype="PCM_16"     
           )
    
    def _set_speech_host_pos(self,listdata,split_size=100):
        '''
            set host moving pos 
            host moves to next pos after several signals 
            src_height: randomly from 1.6 to 1.8
            start_pos: start_pos of moving
            end_pos: end_pos of moving
        '''
        src_height = round(random.uniform(1.6,1.8),2)


        start_pos = [random.uniform(self.room_size[0] - 3*self.d_wall,self.room_size[0] - 2*self.d_wall),random.uniform(self.room_size[1]/2+self.d_wall/2,self.room_size[1]-self.d_wall),src_height]
        end_pos = [random.uniform(self.room_size[0] - 3*self.d_wall,self.room_size[0] - 2*self.d_wall),random.uniform(self.d_wall,self.room_size[1]/2-self.d_wall/2),src_height]

        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        num_points = int(math.sqrt(len(listdata["start_time"])))
        
        # compute each point_pos
        alphas = np.linspace(0, 1, num_points)
        pos_src = np.array([start_pos + alpha * (end_pos - start_pos) for alpha in alphas])
        pos_src = [start_pos + alpha * (end_pos - start_pos) for alpha in alphas]

        
        signal_gain = random.uniform(self.signal_gains_arr[0],self.signal_gains_arr[1])
        self.gains.append(signal_gain)

        file_paths = listdata["file_path"]

        # compute signal_num for each point
        avg = len(listdata["start_time"]) // num_points
        remainder = len(listdata["start_time"]) % num_points
        start_index = 0
        for i in range(num_points):
            chunks = avg + (1 if i <remainder else 0)
            
            for index in range(chunks):
                audio_path = file_paths[start_index + index]
                audio, fs = sf.read(audio_path)
                if fs != self.fs:
                    audio = self.resample(audio,fs,self.fs)
                    fs = self.fs
                    
                if len(audio.shape) > 1:
                    audio = audio[0]
            
                st_time = int(listdata["start_time"][start_index + index] * self.fs)
                ed_time = int(listdata["end_time"][start_index + index] * self.fs)
                audio = audio[:ed_time-st_time]    
                    
                    
                
                if index == 0:
                    final_audio = audio
                else:
                    # merge audio
                    start = listdata["start_time"][start_index + index]
                    tmp = round((start-end_time)*self.fs)
                    audio_pad = np.pad(audio, (tmp, 0), mode='constant')
                    final_audio = np.concatenate([final_audio, audio_pad])

                end_time = listdata["end_time"][start_index + index]
            
            x,y,z = pos_src[i]
            src_pos = [x,y,z]
            self.host_pos.append(src_pos)
            self.host_audio.append(final_audio)
            delay=listdata["start_time"][start_index]
            self.host_delay.append(delay)
            #self.room.add_source(src_pos, signal=final_audio, delay=delay)

            start_index += chunks
    

    def _merge_audio(self, listdata):
        '''
            merge signals for one src
        '''
        file_paths = listdata["file_path"]
        start_time = 0
        
        for index, path in enumerate(file_paths):
            audio, fs = sf.read(path)
            if fs != self.fs:
                audio = self.resample(audio, fs, self.fs)
                fs = self.fs
                
            if len(audio.shape) > 1:
                audio = audio[0]
            
            st_time = int(listdata["start_time"][index] * self.fs)
            ed_time = int(listdata["end_time"][index] * self.fs)
            audio = audio[:ed_time-st_time]
            
            # print
            if index == 0:
                final_audio = audio
            else:
                if(listdata["start_time"][index] < start_time):
                    raise ValueError("Invalid timestamp found")
                
                start = listdata["start_time"][index]
                tmp = (start - start_time) * fs
                audio = np.pad(audio, (round(tmp), 0), mode='constant')
                final_audio = np.concatenate([final_audio, audio])
                
            start_time = listdata["end_time"][index]
        return final_audio
    
    
    def _set_pos(self):
        '''
            set src pos for different meeting type
        '''
        if self.meeting_type == 'circle':
            return self._set_pos_circle()
        elif self.meeting_type == 'desk':
            return self._set_pos_table()
        elif self.meeting_type == 'speech':
            return self._set_pos_speech()

    def _set_pos_circle(self):
        '''
            set src pos for "circle" meeting type
        '''
        attempts = 0
        valid = True
        src_height = round(random.uniform(1.2, 1.4),3)
        while attempts < self.max_attempts:
            valid = True
            angle = random.uniform(0, 2 * math.pi)
            for exisit in self.angles:
                angle_new = min(abs(angle - exisit),2 * math.pi - abs(angle - exisit))
                if angle_new < self.min_angle:
                    valid = False
                    break
            if valid:  break
            attempts += 1
            
        x = self.src_center[0] + self.radius * math.cos(angle)
        y = self.src_center[1] + self.radius * math.sin(angle)
        self.angles.append(angle)
        self.loc.append((x, y, src_height))
        # print(f"[Warning] Forced placement at ({x:.2f}, {y:.2f}) after {attempts} attempts.")
        return [x,y,src_height]

    def _set_pos_table(self):
        '''
            set src pos for "desk" meeting type
        '''
        src_height = round(random.uniform(1.2, 1.4), 3)

        def top_edge(): return (round(random.uniform(self.d_wall, self.width), 2), self.length)
        def right_edge(): return (self.width, round(random.uniform(self.d_wall, self.length), 2))
        def bottom_edge(): return (round(random.uniform(self.d_wall, self.width), 2), self.d_wall)
        def left_edge(): return (self.d_wall, round(random.uniform(self.d_wall, self.length), 2))

        edges = [top_edge, right_edge, bottom_edge, left_edge]

        def is_valid_pos(x1, y1):
            return all(math.hypot(x1 - x, y1 - y) >= self.d for x, y, _ in self.loc)

        random.shuffle(edges)
        attempts = 0
        final_pos = None
        while attempts < self.max_attempts:
            for edge in edges:
                x, y = edge()
                if is_valid_pos(x, y):
                    self.loc.append((x, y, src_height))
                    return [x, y, src_height]
                final_pos = (x, y)
            attempts += 1

        x, y = final_pos
        self.loc.append((x, y, src_height))
        # print(f"[Warning] Forced placement at ({x:.2f}, {y:.2f}) after {attempts} attempts.")
        return [x, y, src_height]

    def _set_pos_speech(self):
        '''
            set audience pos for "speech" meeting type
        '''
        src_height = round(random.uniform(1.2, 1.4), 3)

        attempts = 0
        final_pos = None

        while attempts < self.max_attempts:
            r = random.uniform(0, self.max_radius)
            theta = random.uniform(0, 2 * math.pi)

            x = self.src_center[0] + r * math.cos(theta)
            y = self.src_center[1] + r * math.sin(theta)

            is_valid = all(
                math.hypot(x - ex, y - ey) >= self.d
                for ex, ey, _ in self.loc
            )

            if is_valid:
                self.loc.append((x, y, src_height))
                return [x, y, src_height]

            final_pos = (x, y)
            attempts += 1

        x, y = final_pos
        self.loc.append((x, y, src_height))
        # print(f"[Warning] Forced speech source placement at ({x:.2f}, {y:.2f}) after {attempts} attempts.")
        return [x, y, src_height]

    def _set_gain(self, signal, target_db):
        '''
            set signal gain
            signal: The input audio signal (Tensor)
            target_db: target gain
            return: target audio signal (Tensor)
        '''

        def compute_rms(wav):
            return torch.sqrt(torch.mean(wav**2) + 1e-8)

        def normalize_rms(wav, target_rms=0.01):
            rms = compute_rms(wav)
            return wav * (target_rms / rms)

        def apply_gain_with_rms(wav, gain_db, target_rms=0.01):
            wav = normalize_rms(wav, target_rms)
            scale = 10 ** (gain_db / 20)
            return wav * scale

        gain_applied_wav = apply_gain_with_rms(signal,target_db, target_rms=0.01)

        return gain_applied_wav



    def resample(self,y, original_sample_rate, target_sample_rate: int = 16_000):
        return signal.resample(y, int(len(y) * target_sample_rate / original_sample_rate))

    def _point_noise_simulate(self,noise,start,noise_list):
        '''
            simulate each point noise
        '''
        room_for_noise = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            materials=pra.Material(self.e_absorption),
            max_order=self.max_order,
            ray_tracing=True,
            air_absorption=True,
        )

        room_for_noise.add_microphone(self.mic_loc)

        noise_height = round(random.uniform(1.2,1.4),3)

        #pos
        if noise_list[1] == 'near':
            center = self.loc[random.randint(0,len(self.loc)-1)]
            r = random.uniform(0.1,0.35)
            theta = random.uniform(0, 2 * math.pi)
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)

            # Prevent out-of-bounds and ensure that it is within [d, room_size - d]
            x = min(max(x, self.d ), self.room_size[0] - self.d)
            y = min(max(y, self.d), self.room_size[1] - self.d)

            noise_pos = [x,y,noise_height]


        elif noise_list[1] == 'far':
            width = self.room_size[0]-self.d
            length = self.room_size[1]-self.d
            def top_edge():
                return (round(random.uniform(self.d, width),2), length)
                
            def bottom_edge():
                return (round(random.uniform(self.d, width),2), self.d)
                
            def left_edge():
                return (self.d, round(random.uniform(self.d, length),2))
            
            edges = [top_edge, bottom_edge, left_edge]
            random.shuffle(edges)

            for edge in edges:
                x,y = edge()
                break

            noise_pos  = [x,y,noise_height]

        
        tmp = 5*self.fs
        if noise_list[0] == 'music':
            room_for_noise.add_source(noise_pos, signal=noise[tmp:tmp+min(tmp,len(noise))], delay = start)
        else:
            room_for_noise.add_source(noise_pos, signal=noise[:min(tmp,len(noise))], delay = start)

        room_for_noise.simulate()
        
        return room_for_noise.mic_array.signals

    def _gen_point_noise(self,category_files,target_categories,noise_path,min_segments = 15,max_segments = None):
        '''
            gen final point noise audio
            gen intervals between each point noise
            merge point noise audio
        '''

        if max_segments == None:
            max_segments = int(self.audio_len/16)
        point = np.random.randint(min_segments, max_segments)

        intervals = np.random.exponential(15.0, size=point)
        interval = np.clip(intervals,8,None)
        time_cursor = 0.0
        noise_num = 0
        final_point_noise_audio = None
        for i in range(point):
            noise_list = random.choice(target_categories)#type pos
            filename = random.choice(category_files[noise_list[0]])
            
            point_noise_path = os.path.join(noise_path,filename)


            noise, fs = sf.read(point_noise_path)
            if fs != self.fs:
                noise = self.resample(noise,fs,self.fs)
                fs = self.fs
            #noise = (self.set_gain(torch.from_numpy(noise),-50)).numpy()

            dur = min(len(noise)/fs,5)
            start = time_cursor + interval[i]
            end = start + dur
            
            if end > self.audio_len or i == point-1:
                # print(f"add {noise_num} point noise")
                break 
            self.point_noise_time += dur 
            time_cursor = end

            point_noise_audio = self._point_noise_simulate(noise,interval[i],noise_list)

            if final_point_noise_audio is None:
                final_point_noise_audio = point_noise_audio
            else:
                final_point_noise_audio = np.concatenate([final_point_noise_audio, point_noise_audio],axis = 1)

            noise_num+=1
            
        return final_point_noise_audio

    def _compute_DRR(self,h, fs, t_direct_ms=5):
        '''
            compute DRR
        '''
        t_direct = int(fs * t_direct_ms / 1000)
        peak_idx = np.argmax(np.abs(h))
        direct_part = h[peak_idx : peak_idx + t_direct]
        reverb_part = np.concatenate([h[:peak_idx], h[peak_idx + t_direct:]])
        
        energy_direct = np.sum(direct_part ** 2)
        energy_reverb = np.sum(reverb_part ** 2)
        drr_db = 10 * np.log10(energy_direct / (energy_reverb + 1e-12))
        return drr_db
    
    def _compute_SRR(self,s, h, fs, t_early_ms=50):
        '''
            compute SRR
        '''
        h_early = np.copy(h)
        h_late = np.copy(h)
        t_early = int(fs * t_early_ms / 1000)
        h_early[t_early:] = 0
        h_late[:t_early] = 0
        
        x_early = convolve(s, h_early)
        x_late = convolve(s, h_late)
        
        energy_early = np.sum(x_early ** 2)
        energy_late = np.sum(x_late ** 2)
        srr_db = 10 * np.log10(energy_early / (energy_late + 1e-12))
        return srr_db
    
    

    def _add_noise(self,speech_sig, vad_duration, point_noise,diffuse_noise, SNR_point,SNR_diffuse):
        '''
            add noise to the audio.
            speech_sig: The input audio signal (Tensor).
            vad_duration: The length of the human voice (int).
            noise_sig: The input noise signal (Tensor).
            snr: the SNR you want to add (int).
            returns: noisy speech sig with specific snr.
        '''

        if vad_duration != 0:
            snr1 = 10**(SNR_point/10.0)
            snr2 = 10**(SNR_diffuse/10.0)
            speech_power = torch.sum(speech_sig**2)/vad_duration

            point_power = torch.sum(point_noise**2)/int(self.point_noise_time)
            point_update =  point_noise / torch.sqrt(snr1 * point_power/speech_power)

            diffuse_power = torch.sum(diffuse_noise**2)/int(diffuse_noise.shape[0]/self.fs)
            diffuse_update = diffuse_noise / torch.sqrt(snr2 * diffuse_power/speech_power)
            
            avg_diffuse_snr = (self.audio_len-self.point_noise_time)*SNR_diffuse 
            diifuse_point_snr =  speech_power / (point_power+diffuse_power)
            avg_diifuse_point_snr = self.point_noise_time * 10 * torch.log10(diifuse_point_snr).numpy()
            
            self.avg_snr= (avg_diffuse_snr+avg_diifuse_point_snr)/self.audio_len
            
            
            def adjust_noise_length(noise, target_length):
                '''
                    adjust_noise_length
                '''
                if target_length > noise.shape[0]:
                    # repeat
                    repeat_times = int(np.ceil(target_length / noise.shape[0]))
                    repeated_noise = noise.repeat(repeat_times+1,1)
                    return repeated_noise[:target_length,:]
                else:
                    # cut
                    return noise[ :target_length,:]
                
            point_noise_sig = adjust_noise_length(point_update, speech_sig.shape[0])

            diffuse_noise_sig = adjust_noise_length(diffuse_update, speech_sig.shape[0])

            return speech_sig  + point_noise_sig + diffuse_noise_sig
        
        else:
            return speech_sig

    
    def add_noise(self, 
                  output_dir, 
                  point_noise_path: str, 
                  diffuse_noise_path: str
                  ):
        '''
            add noise to clean audio
        '''

        SNR_point = self.simulate_config.get('SNR_point', )
        SNR_point_arr = self.simulate_config.get('SNR_point_arr', )
        SNR_diffuse = self.simulate_config.get('SNR_diffuse', )  
        SNR_diffuse_arr = self.simulate_config.get('SNR_diffuse_arr', )    
        
        
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        # if os.path.exists(output_path_final):
        #     return 
        
        point_noise = self._gen_point_noise(self.category_files, self.target_categories, point_noise_path)
        diffuse_file = random.choice(os.listdir(diffuse_noise_path))
        diffuse_noise, fs = sf.read(os.path.join(diffuse_noise_path, diffuse_file))
        
        
        if fs != self.fs:
            noise = self.resample(noise, fs, self.fs)
            fs = self.fs
        
        diffuse_noise = diffuse_noise[:,np.newaxis]
        if SNR_point == None:
            SNR_point = random.randint(SNR_point_arr[0],SNR_point_arr[1])
        self.SNR_point = SNR_point

        if SNR_diffuse == None:
            SNR_diffuse = random.randint(SNR_diffuse_arr[0],SNR_diffuse_arr[1])
        self.SNR_diffuse = SNR_diffuse


        reverb_signal = self.room.mic_array.signals
        final_signal = self._add_noise(speech_sig = torch.from_numpy(reverb_signal.T),
                                       vad_duration = self.vad_dur,
                                       point_noise = torch.from_numpy(point_noise.T),
                                       diffuse_noise = torch.from_numpy(diffuse_noise), 
                                       SNR_point = self.SNR_point,
                                       SNR_diffuse = self.SNR_diffuse
                                    )
        
        final_signal = final_signal.numpy()
        os.makedirs(os.path.join(output_dir,"noisy"), exist_ok=True)  
        output_path_noisy = os.path.join(output_dir,"noisy", f"{filename}.wav")
        sf.write(
            output_path_noisy,  
            final_signal,         
            self.fs,                   
            subtype="PCM_16"      
        )        

        info_path = f"{filename}_info.json"
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)  

        with open(os.path.join(json_dir, info_path), "w", encoding="utf-8") as file:
            file.write(self.to_json())



if __name__ == '__main__':
    
    import yaml
    config = yaml.safe_load(open("/home3/yihao/Research/Code/Large-scale-diarization-dataset/config/config.yaml", 'r'))
    simulate_config = config["simulate_config"]
    random.seed(42)
    room = singlechannel_rir_room(
        filepath = "/home3/yihao/Research/Code/Large-scale-diarization-dataset/exp/exp1/test/samples/00_00000_pre.list",
        simulate_config = simulate_config
    )
    room.simulate("/home3/yihao/Research/Code/Large-scale-diarization-dataset/exp/exp1/test/wavs")
    
    

    room.add_noise("/home3/yihao/Research/Code/Large-scale-diarization-dataset/exp/exp1/test/wavs",
                   "/home3/yihao/Research/Code/Large-scale-diarization-dataset/noise_dataset/point_noise",
                   "/home3/yihao/Research/Code/Large-scale-diarization-dataset/noise_dataset/diffuse_noise"
                   )
    



