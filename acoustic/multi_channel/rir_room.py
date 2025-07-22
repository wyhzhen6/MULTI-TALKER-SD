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
import scipy
from scipy.signal import stft, istft,get_window
from scipy.linalg import cholesky, eigh
from scipy.signal import convolve
from collections import defaultdict


class rir_room:
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
        self.mic_radius = simulate_config.get('mic_radius')
        self.mic_radius_arr = simulate_config.get('mic_radius_arr')
        self.mic_num = simulate_config.get('mic_num')
        self.mic_num_arr = simulate_config.get('mic_num_arr')
        self.array_num = simulate_config.get('array_num')
        self.linear_array_mic_num = simulate_config.get('linear_array_mic_num')
        self.linear_array_distances = simulate_config.get('linear_array_distances')
        self.is_compute_DRR = simulate_config['is_compute_DRR']
        self.is_compute_SRR = simulate_config['is_compute_SRR']
        self.mic_center_loc = simulate_config['mic_center_pos']
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
        self.d = 0.5        # min_dis
        self.d_src = round(random.uniform(0.4, 0.5), 2) # min_src_dis
        self.d_wall = round(random.uniform(0.45, 0.55), 2) # min_src_wall

        # setting rir parameters
        self.mic_loc = []
        self.mic_center_loc = []
        self._generate_src_pra()
        self._create_mic()
        
        
        # prepare noise type, path
        self._prepare_point_noise()

        # others
        self.max_attempts = 10000
        self.angles = []
        self.loc = []
        self.DRR = []
        self.SRR = []
        self.SRR_circle = []
        self.DRR_circle = []
        self.SRR_linear = []
        self.DRR_linear = []
        self.host_audio = []
        self.gains = []

        self.host_pos = []
        self.drr = 0
        self.srr = 0

        # noise
        self.SNR_point = 0
        self.SNR_diffuse = 0
        self.point_noise_time = 0
        self.avg_snr = []
        
        
        
        self.vad_label = []
        
        
        

    def to_dic(self):
        return {
            "meeting_type" : self.meeting_type,
            "room_size" : self.room_size,
            "room_type" :self.room_type,
            "rt60 " :self.rt60,
            "fs " :self.fs,
            "array_num " :self.array_num,
            "circle array's mic_num" :self.mic_num,
            "circle array's center_pos" :self.mic_center_loc,
            "linear array's mic_num" :self.linear_array_mic_num,
            "linear array's center_pos" :self.linear_array_center,
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
            #start_time = float(parts[0])
            #end_time = float(parts[1])
            if len(parts) >= 4:
                start_time = float(parts[0])
                end_time = float(parts[1])

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


    def _generate_room_pra(self,meeting_types):
        '''
            generate parameters for rir:
            rt60,
            room_size,
            e_absorption,
            max_order,
            meeting_type
        '''
        if self.rt60 == None:
            if len(self.listdata)<20:
                self.rt60 = round(random.uniform(self.rt60_mid[0], self.rt60_mid[1]),2)
            else:
                self.rt60 = round(random.uniform(self.rt60_lar[0], self.rt60_lar[1]),2)
                
        if self.room_size == None:
            if len(self.listdata)<20:
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
            meeting_types = meeting_types if len(self.listdata) < 20 else [meeting_types[0], meeting_types[2]]
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
            self.min_angle = 2 * math.asin(self.d_wall / (2 * self.radius))
        elif self.meeting_type == "desk":
            self.width = self.room_size[0]-2*self.d_wall
            self.length = self.room_size[1]-self.d_wall
        elif self.meeting_type == "speech":
            self.src_center = [(self.room_size[0]-4*self.d_wall)/2,self.room_size[1]/2]
            self.max_radius = min(self.src_center[0],self.src_center[1])-self.d_wall
         
    def _create_mic(self):

        mic_height =[round(random.uniform(0.8,1),2),round(random.uniform(self.room_size[2]-1,self.room_size[2]-self.d),2)]#desk,ceiling
        
        if self.mic_num == None:
            self.mic_num = self.mic_num_arr[random.randint(0,1)]
        if self.mic_radius == None:
            self.mic_radius = self.mic_radius_arr[random.randint(0,len(self.mic_radius_arr)-1)]
        if self.array_num == None:
            self.array_num =  1 if len(self.listdata) < 20 else 2
            self.SRR_circle2 = []
            self.DRR_circle2 = []   


        #add circle mic array
        for i in range(self.array_num):
            if self.meeting_type == 'desk':
                self.mic_center_pos = [self.d+(i+1)*(self.width-self.d)/(self.array_num+1),self.room_size[1]/2]
                h=mic_height[0]#desk
            elif self.meeting_type == 'circle':
                self.mic_center_pos = [self.src_center[0]-self.max_radius+(i+1)*2*self.max_radius/(self.array_num+1),self.room_size[1]/2]
                h = mic_height[0]#desk
            elif self.meeting_type == 'speech':
                self.mic_center_pos = [self.src_center[0]-self.max_radius+(i+1)*2*self.max_radius/(self.array_num+1),self.room_size[1]/2]
                h = mic_height[1]#ceiling
        
            R = pra.circular_2D_array(self.mic_center_pos,self.mic_num,0,self.mic_radius)#circular_2D_array(center, M, phi0, radius)
            if self.mic_num == 6:
                new_column = np.array(self.mic_center_pos).reshape(-1, 1)  
                R = np.hstack((R, new_column))#6+1

            z_coords = np.full(R.shape[1], h)  
            R = np.vstack([R, z_coords])
            self.mic_loc.append(R)
            self.mic_center_loc.append(self.mic_center_pos)

            self.room.add_microphone_array(R)

        #add linear mic array
        h = round(random.uniform(1.6,1.8),2)
        d = 0.05
        self.linear_array_center = [self.room_size[0]-self.d,self.room_size[1]/2]
        self.mic_center_loc.append(self.linear_array_center)

        #generate fixed distance 
        linear_mic_pos = self.random_linear_array(self.linear_array_center)
        z_coords = np.full(linear_mic_pos.shape[1], h)  
        linear_mic_pos = np.vstack([linear_mic_pos, z_coords])
        
        self.mic_loc.append(linear_mic_pos)
        self.room.add_microphone_array(linear_mic_pos)

    def unit_vec2D(self,phi):
        return np.array([[np.cos(phi), np.sin(phi)]]).T

    def random_linear_array(self,center, phi=np.pi/2 , M = 8):

        u = self.unit_vec2D(phi)
        

        cumulative_distances = np.cumsum(np.insert(self.linear_array_distances, 0, 0))
        
        centered_distances = cumulative_distances - cumulative_distances.mean()

        positions = np.array(center)[:, np.newaxis] + centered_distances[np.newaxis, :] * u
        return positions
    

    def _set_host_label(self,host_label):
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

        # set host label
        if self.meeting_type == "speech":
            print("speech")
            host_label = self.simulate_config.get('host_label', )
            self._set_host_label(host_label)

        audio = []
        for index, item in enumerate(self.listdata):
            if item["label"] != self.speech_host_label:
                #set src pos
                src_pos = self._set_pos()
                # merge audio
                signals = self._merge_audio(item)

                # set gain
                signal_gain = random.uniform(self.signal_gains_arr[0],self.signal_gains_arr[1])
                self.gains.append(signal_gain)
                signals = self._set_gain(torch.from_numpy(signals),signal_gain).numpy()
                
                # prepare audio for computing SRR
                if self.is_compute_SRR:
                    audio.append(signals)
                
                # add source for rir room
                self.room.add_source(src_pos, signal=signals, delay=item["start_time"][0])
                
                '''if room.is_compute_DRR:
                    #compute DRR
                    room.DRR.append(room.compute_DRR(signals,room.fs))'''
        
        if self.meeting_type == "speech":
            item = next((item for item in self.listdata if item["label"] == self.speech_host_label), None)
            self.set_speech_host_pos(item)

        # rir simulate
        self.room.simulate()
        
        # prepare rir for computing SRR, DRR
        if self.is_compute_SRR or self.is_compute_DRR:
            for i in range(len(audio)):
                rir1 = self.room.rir[0][i]
                rir2 = self.room.rir[len(self.room.rir)-1][i]
                self.SRR_circle.append(self.compute_SRR(audio[i],rir1,self.fs))
                self.SRR_linear.append(self.compute_SRR(audio[i],rir2,self.fs))

                self.DRR_circle.append(self.compute_DRR(rir1,self.fs))
                self.SRR_linear.append(self.compute_DRR(rir2,self.fs))

                if self.array_num == 2:
                    rir3 = self.room.rir[self.mic_num+1][i]
                    self.SRR_circle2.append(self.compute_SRR(audio[i],rir3,self.fs))
                    self.DRR_circle2.append(self.compute_DRR(rir3,self.fs))

            if self.meeting_type == "speech":
                
                item = next((item for item in self.listdata if item["label"] == self.speech_host_label), None)
                
                host_srr_circle = []
                host_srr_circle2 = []
                host_srr_linear = []
                host_drr_circle = []
                host_drr_circle2 = []
                host_drr_linear = []
                
                for index2,audio_host in enumerate(self.host_audio):

                    rir1 = self.room.rir[0][index2+len(audio)] 
                    rir_linear = self.room.rir[len(self.room.rir)-1][index2 + len(audio)]

                    host_srr_circle.append(self.compute_SRR(audio_host,rir1,self.fs))
                    host_srr_linear.append(self.compute_SRR(audio_host,rir_linear,self.fs))
                    
                    host_drr_circle.append(self.compute_DRR(rir1,self.fs))
                    host_drr_linear.append(self.compute_DRR(rir_linear,self.fs))

                    if self.array_num == 2:
                        rir2 = self.room.rir[self.mic_num+1][index2+len(audio)]
                        host_srr_circle2.append(self.compute_SRR(audio_host,rir2,self.fs))
                        host_drr_circle2.append(self.compute_DRR(rir2,self.fs))
                
                self.SRR_circle.append(np.mean(host_srr_circle))
                self.SRR_linear.append(np.mean(host_srr_linear))

                self.DRR_circle.append(np.mean(host_drr_circle))
                self.DRR_linear.append(np.mean(host_drr_linear))

                if self.array_num == 2:
                    self.SRR_circle2.append(np.mean(host_srr_circle2))
                    self.DRR_circle2.append(np.mean(host_drr_circle2))

        if self.is_compute_SRR:      
            tmp =  np.mean(self.SRR_circle)         
            self.SRR.append(tmp.tolist()) 

            if self.array_num == 2:
                tmp =  np.mean(self.SRR_circle2)
                self.SRR.append(tmp.tolist())

            tmp =  np.mean(self.SRR_linear)
            self.SRR.append(tmp.tolist())

            self.srr = np.mean(self.SRR)
            

        if self.is_compute_DRR:
            tmp =  np.mean(self.DRR_circle)         
            self.DRR.append(tmp.tolist()) 

            if self.array_num == 2:
                tmp =  np.mean(self.DRR_circle2)
                self.DRR.append(tmp.tolist())

            tmp =  np.mean(self.DRR_linear)
            self.DRR.append(tmp.tolist())

            self.drr = np.mean(self.DRR)
        
        #save clean audio
        arraynum = 0
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
    
        #os.makedirs(os.path.join(output_dir,"clean"))
        #os.makedirs(os.path.join(output_dir,"noisy"))

        for i in range(len(self.mic_loc)):
            array_signal = self.room.mic_array.signals[arraynum : arraynum + len(self.mic_loc[i][1])]
            arraynum += len(self.mic_loc[i][1])
            
            output_path_clean = os.path.join(output_dir,"clean", f"array_{i+1}_clean_signal.wav")
            sf.write(
                output_path_clean,  
                array_signal.T,         
                self.fs,                   
                subtype="PCM_16"     
            )
            
    def set_speech_host_pos(self,listdata,split_size=100):
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

        dist_total = norm(start_pos - end_pos)
        unit_vec = (start_pos - end_pos) / dist_total
        
        num_points = int(math.sqrt(len(listdata["start_time"])))
        
        # generate each point_pos
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
            
                st_time = int(listdata["start_time"][index] * self.fs)
                ed_time = int(listdata["end_time"][index] * self.fs)
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
            self.room.add_source(src_pos, signal=final_audio, delay=delay)

            start_index += chunks


    def _merge_audio(self,listdata):

        file_paths = listdata["file_path"]
        start_time = 0
        
        for index, path in enumerate(file_paths):
            audio, fs = sf.read(path)
            if fs != self.fs:
                audio = self.resample(audio,fs,self.fs)
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
                    continue
                
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
        src_height = round(random.uniform(1.2,1.4),3)
        while attempts < self.max_attempts:
            valid = True
            angle = random.uniform(0, 2 * math.pi)
            for exisit in self.angles:
                angle_new = min(abs(angle - exisit),2 * math.pi - abs(angle - exisit))
                if angle_new < self.min_angle:
                    valid = False
                    break
            if valid:
                x = self.src_center[0] + self.radius * math.cos(angle)
                y = self.src_center[1] + self.radius * math.sin(angle)
                self.angles.append(angle)
                self.loc.append((x, y,src_height))
                break

            attempts += 1
        if not valid:
            raise ValueError("no available pos")

        return [x,y,src_height]

    def _set_pos_table(self):
        '''
            set src pos for "desk" meeting type
        '''
        src_height = round(random.uniform(1.2,1.4),3)
        def top_edge():
            return (round(random.uniform(self.d_wall, self.width),2), self.length)
            
        def right_edge():
            return (self.width, round(random.uniform(self.d_wall, self.length),2))
            
        def bottom_edge():
            return (round(random.uniform(self.d_wall, self.width),2), self.d_wall)
            
        def left_edge():
            return (self.d_wall, round(random.uniform(self.d_wall, self.length),2))
        
        def is_valid_pos(x1,y1):
            for x,y,z in self.loc:
                dis = math.sqrt((x1-x)**2 + (y1-y)**2)
                if dis < self.d:
                    return  False
            return True

        edges = [top_edge, right_edge, bottom_edge, left_edge]

        
        random.shuffle(edges)
        placed = False
        attempts = 0

        while attempts < self.max_attempts:
            for edge in edges:
                x,y = edge()
                if is_valid_pos(x, y):
                    self.loc.append((x, y,src_height))
                    placed = True
                    break
            if placed:
                break
            attempts +=1

        if not placed:
            raise ValueError("no available pos")

        return [x,y,src_height]
    
    def _set_pos_speech(self):
        '''
            set audience pos for "speech" meeting type
        '''
        src_height = round(random.uniform(1.2,1.4),3)

        attempts = 0
        valid = True
        while attempts < self.max_attempts:
            r = random.uniform(0,self.max_radius)
            theta = random.uniform(0, 2 * math.pi)

            x = self.src_center[0] + r * math.cos(theta)
            y = self.src_center[1] + r * math.sin(theta)

            valid = True
            for existing_x, existing_y,z in self.loc:
                distance = math.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
                if distance < self.d:
                    valid = False
                    break
                
            if valid:
                self.loc.append((x, y,src_height))
                break
                
            attempts += 1
            
        if not valid:
            raise ValueError("no available pos")
            
        return [x,y,src_height]

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
    
    def gen_desired_spatial_coherence(self, pos_mics: np.ndarray, fs: int, noise_field: str = 'spherical', c: float = 343.0, nfft: int = 256) -> np.ndarray:
        '''
        generate desired spatial coherence for one array

        Args:
            pos_mics: microphone positions, shape (num_mics, 3)
            fs: sampling frequency
            noise_field: 'spherical' or 'cylindrical'
            c: sound velocity
            nfft: points of fft

        Raises:
            Exception: Unknown noise field if noise_field != 'spherical' and != 'cylindrical'

        Returns:
            np.ndarray: desired spatial coherence, shape [num_mics, num_mics, num_freqs]
            np.ndarray: desired mixing matrices, shape [num_freqs, num_mics, num_mics]


        Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
        '''
        #assert pos_mics.shape[1] == 7, pos_mics.shape

        M = pos_mics.shape[0]
        num_freqs = nfft // 2 + 1

        # compute desired spatial coherence matric
        ww = 2 * math.pi * fs * np.array(list(range(num_freqs))) / nfft
        dist = np.linalg.norm(pos_mics[:, np.newaxis, :] - pos_mics[np.newaxis, :, :], axis=-1, keepdims=True)
        if noise_field == 'spherical':
            DSC = np.sinc(ww * dist / (c * math.pi))
        elif noise_field == 'cylindrical':
            DSC = scipy.special.j0(ww * dist / c)
        else:
            raise Exception('Unknown noise field')

        # compute mixing matrices of the desired spatial coherence matric

        Cs = np.zeros((num_freqs, M, M), dtype=np.complex128)
        for k in range(1, num_freqs):
            
            DSC[:, :, k] = (DSC[:, :, k] + DSC[:, :, k].conj().T) / 2
            
            D, V = eigh(DSC[:, :, k])
            D = np.clip(D, a_min=1e-8, a_max=None)
            C = V @ np.diag(np.sqrt(D))

            Cs[k, ...] = C

        return DSC, Cs


    def prepare_diffuse_noise(self, noise: np.ndarray, L: int, Cs: np.ndarray, nfft: int = 256, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
        '''generate diffuse noise with the mixing matrice of desired spatial coherence

        Args:
            noise: at least `num_mic*L` samples long
            L: the length in samples
            Cs: mixing matrices, shape [num_freqs, num_mics, num_mics]
            nfft: the number of fft points
            rng: random number generator used for reproducibility

        Returns:
            np.ndarray: multi-channel diffuse noise, shape [num_mics, L]
        '''

        M = Cs.shape[-1]

        assert noise.shape[-1] >= M * L, ("The noise signal should be at least `num_mic*L` samples long", noise.shape, M, L)
        start = rng.integers(low=0, high=noise.shape[-1] - M * L + 1)
        noise = noise[start:start + M * L].reshape(M, L)
        noise = noise - np.mean(noise, axis=-1, keepdims=True)


        f, t, N = stft(noise, fs = 16000,window='hann', nperseg=nfft, noverlap=int(0.75 * nfft), nfft=nfft)  # N: [M,F,T]
        # Generate output in the STFT domain for each frequency bin k
        X = np.einsum('fmn,mft->nft', np.conj(Cs), N)
        # Compute inverse STFT
        F, x = istft(X, window='hann', nperseg=nfft, noverlap=0.75 * nfft, nfft=nfft)
        x = x[:, :L]
        return x  # [M, L]
    
    
    def gen_diffuse_noise(self,noise_path,nfft = 1024,fs=16000):
        diffuse_noise = []
        for i in range(len(self.mic_loc)):
            pos_mics = self.mic_loc[i].T.tolist() 

            if i == (len(self.mic_loc) - 1):
                #linear array
                if len(pos_mics) % 2 != 0:
                    center = pos_mics[int(len(pos_mics)/2)]
                else:

                    x = min(pos_mics[int(len(pos_mics)/2)-1][0],pos_mics[int(len(pos_mics)/2)][0])+abs(pos_mics[int(len(pos_mics)/2)-1][0]-pos_mics[int(len(pos_mics)/2)][0])/2
                    y = min(pos_mics[int(len(pos_mics)/2)-1][1],pos_mics[int(len(pos_mics)/2)][1])+abs(pos_mics[int(len(pos_mics)/2)-1][1]-pos_mics[int(len(pos_mics)/2)][1])/2
                    z = min(pos_mics[int(len(pos_mics)/2)-1][2],pos_mics[int(len(pos_mics)/2)][2])+abs(pos_mics[int(len(pos_mics)/2)-1][2]-pos_mics[int(len(pos_mics)/2)][2])/2
                    center = [x,y,z]
            else:
                #circle array
                center = pos_mics[len(pos_mics)-1]


            for i,item in enumerate(pos_mics):
                for index,_ in enumerate(item):
                    item[index] = item[index] - center[index]

            pos_mics = np.array(pos_mics)

            noise, fs = sf.read(noise_path)
            if fs != self.fs:
                noise = self.resample(noise,fs,self.fs)
                fs = self.fs
            #noise = (self._set_gain(torch.from_numpy(noise),-200)).numpy()

            DSC, Cs = self.gen_desired_spatial_coherence(pos_mics=pos_mics, fs=fs, noise_field='spherical', nfft=nfft)
            
            L = int(noise.shape[0] / Cs.shape[1])
            diffuse_noise.append(self.prepare_diffuse_noise(noise=noise, L=L, Cs=Cs, nfft=nfft))
            

        return diffuse_noise


    def point_noise_simulate(self,noise,start,noise_list):
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
        for i,item in enumerate(self.mic_loc):
            room_for_noise.add_microphone_array(item)

        noise_height = round(random.uniform(1.2,1.4),3)

        #pos
        if noise_list[1] == 'near':
            center = self.loc[random.randint(0,len(self.loc)-1)]
            r = random.uniform(0.1,0.5)
            theta = random.uniform(0, 2 * math.pi)
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)

            noise_pos = [x,y,noise_height]


        elif noise_list[1] == 'far':
            width = self.room_size[0]-2*self.d
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

    def gen_point_noise(self,category_files,target_categories,noise_path,min_segments = 15,max_segments = None):
        '''
            gen final point noise audio
            gen intervals between each point noise
            merge point noise audio
        '''
        point_noise = []
        
        if max_segments == None:
            max_segments = int(self.audio_len/16)
        point = np.random.randint(min_segments, max_segments)

        intervals = np.random.exponential(15.0, size=point)
        interval = np.clip(intervals,8,None)
        time_cursor = 0.0
        noise_num = 0
        final_point_noise_audio = None
        for i in range(point):
            noise_list = random.choice(self.target_categories)#type pos
            filename = random.choice(self.category_files[noise_list[0]])
            
            point_noise_path = os.path.join(noise_path,filename)


            noise, fs = sf.read(point_noise_path)
            if fs != self.fs:
                noise = self.resample(noise,fs,self.fs)
                fs = self.fs
            #noise = (self._set_gain(torch.from_numpy(noise),-50)).numpy()

            dur = min(len(noise)/fs,5)
            start = time_cursor + interval[i]
            end = start + dur
            self.point_noise_time += dur
            if end > self.audio_len or i == point-1:
                print(f"add {noise_num} point noise")
                break  
            time_cursor = end
            point_noise_audio = self.point_noise_simulate(noise,interval[i],noise_list)

            if i == 0:
                final_point_noise_audio = point_noise_audio
            else:
                final_point_noise_audio = np.concatenate([final_point_noise_audio, point_noise_audio], axis=1)

            noise_num+=1

        array_num = 0
        for i in range(len(self.mic_loc)):
            array_signal = final_point_noise_audio[array_num : array_num + len(self.mic_loc[i][1])]
            array_num += len(self.mic_loc[i][1])
            
            point_noise.append(array_signal)
        
        return point_noise

    def compute_DRR(self,h, fs, t_direct_ms=5):
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
    
    def compute_SRR(self,s, h, fs, t_early_ms=50):
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


    def _add_noise(self, speech_sig, vad_duration, point_noise, diffuse_noise, SNR_point, SNR_diffuse):
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
            speech_power = torch.sum(speech_sig**2,dim=0, keepdim=True)/vad_duration

            point_power = torch.sum(point_noise**2,dim=0, keepdim=True)/int(self.point_noise_time)
            point_update =  point_noise / torch.sqrt(snr1 * point_power/speech_power)

            diffuse_power = torch.sum(diffuse_noise**2,dim=0, keepdim=True)/int(diffuse_noise.shape[0]/self.fs)
            diffuse_update = diffuse_noise / torch.sqrt(snr2 * diffuse_power/speech_power)
            
            avg_diffuse_snr = (self.audio_len-self.point_noise_time)*SNR_diffuse 
            diifuse_point_snr =  speech_power / (point_power+diffuse_power)
            avg_diifuse_point_snr = self.point_noise_time * 10 * torch.log10(diifuse_point_snr).numpy()
            
            self.avg_snr.append( (avg_diffuse_snr+avg_diifuse_point_snr)/self.audio_len)
            

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

            return speech_sig + diffuse_noise_sig + point_noise_sig
        
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
        
        SNR_point             = simulate_config.get('SNR_point', )
        SNR_point_arr         = simulate_config.get('SNR_point_arr', )
        SNR_diffuse           = simulate_config.get('SNR_diffuse', )    
        SNR_diffuse_arr       = simulate_config.get('SNR_diffuse_arr', )
        
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        
        
        #gen point_noise
        point_noise = self.gen_point_noise(self.category_files,self.target_categories,point_noise_path)
        
        diffuse_file = random.choice(os.listdir(diffuse_noise_path))
        diffuse_noise = self.gen_diffuse_noise(os.path.join(diffuse_noise_path,diffuse_file))
        
        if SNR_point == None:
            SNR_point = random.randint(SNR_point_arr[0],SNR_point_arr[1])
        self.SNR_point = SNR_point

        if SNR_diffuse == None:
            SNR_diffuse = random.randint(SNR_diffuse_arr[0],SNR_diffuse_arr[1])
        self.SNR_diffuse = SNR_diffuse
        
        arraynum = 0
        for i in range(len(self.mic_loc)):
            clean_signal = self.room.mic_array.signals[arraynum : arraynum + len(self.mic_loc[i][1])]
            arraynum += len(self.mic_loc[i][1])
            final_signal = self._add_noise(speech_sig = torch.from_numpy(clean_signal.T),
                                           vad_duration = self.vad_dur,
                                           point_noise = torch.from_numpy(point_noise[i].T),
                                           diffuse_noise = torch.from_numpy(diffuse_noise[i].T), 
                                           SNR_point = self.SNR_point,
                                           SNR_diffuse = self.SNR_diffuse
                                        )
            
            final_signal = final_signal.numpy()
            output_path_noisy = os.path.join(output_dir,"noisy", f"{filename}_array_{i+1}_noisy_signal.wav")
            sf.write(
                output_path_noisy,  
                final_signal,         
                self.fs,                   
                subtype="PCM_16"      
            )
        snr_avg = []
        for item in self.avg_snr:
            snr_avg.append(np.mean(item))
        self.avg_snr = snr_avg
        

        info_path = f"{filename}_info.json"
        with open(os.path.join(output_dir,info_path), "w", encoding="utf-8") as file:
            file.write(self.to_json())

        


if __name__ == '__main__':
    
    import yaml
    config = yaml.safe_load(open("/data/zjj/rir/rir_test/script_test/config.yaml", 'r'))
    simulate_config = config["multi_channel_simulate_config"]
    random.seed(42)
    room = rir_room(
        filepath = "/data/zjj/rir/rir_single/list/0000057.list",
        simulate_config = simulate_config
    )
    room.simulate("/data/zjj/rir/rir_single/output/0000057")
    
    

    room.add_noise("/data/zjj/rir/rir_single/output/0000057",
                   "/data/zjj/rir/esc50/ESC-50-master/audio",
                   "/data/zjj/rir/noise_test"
                   )

    
