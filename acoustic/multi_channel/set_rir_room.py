import argparse
import os
import subprocess
import re
import soundfile as sf
import json
#import pandas as pd
import numpy as np
import csv
import pandas as pd


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pyroomacoustics as pra
#from pyroomacoustics.directivities import Omnidirectional,Cardioid, DirectionVector
import time

import math
import random
from typing import List, Tuple
from numpy.linalg import norm
import torch

from gen_noise import gen_diffuse_noise,gen_desired_spatial_coherence
from scipy.signal import convolve
from collections import defaultdict


#loc = set_pos_discuss()
class rir_room:
    def __init__(self ,wav_path, list_path,filepath,room_type = None,room_size = None ,fs = 16000,rt60 = None, mic_radius = None, mic_num = None,array_num = None,meeting_type = None,speech_host_label = None,is_compute_DRR = True,is_compute_SRR = True):
        
        self.wav_path = wav_path
        self.meeting_type = meeting_type
        self.room_size = room_size
        self.room_type = room_type
        self.rt60 = rt60
        self.mic_radius = mic_radius
        self.audio_len = 0.0
        self.vad_dur = 0
        self.fs = fs
        self.drr = 0
        self.mic_num = mic_num
        self.linear_array_mic_num = 8
        self.array_num = array_num
        self.speech_host_label = speech_host_label
        self.is_compute_DRR = is_compute_DRR
        self.is_compute_SRR = is_compute_SRR
        self.SNR_point = 0
        self.SNR_diffuse = 0
        self.point_noise_time = 0
        self.max_attempts = 10000
        self.d = 0.5#min_src_dis
        self.d_wall = round(random.uniform(0.5,0.7),2)#min_dis_to_wall
        self.noise_d = round(random.uniform(0.2,0.5),2)#min_dis_to_src
        self.host_audio = []
        self.avg_snr = []
        self.vad_label = []
        self.gains = []
        self.noise_loc = []
        self.mic_loc = []
        self.mic_center_loc = []
        self.listdata = []
        self.loc = []#src_loc
        self.angles = []#src_angles circle
        self.DRR = []
        self.SRR = []
        self.SRR_circle = []
        self.SRR_linear = []
        self.host_pos = []
        self.linear_array_distances = [0.15,0.1,0.05,0.2,0.05,0.1,0.15]

        mic_num = [4,6]
        mic_radius = [0.0425,0.05]
        meeting_type = ["desk","circle","speech"]

        self.read_listfile2(list_path,filepath)
        self.generate_room_pra(meeting_type)
        self.create_room()
        self.generate_src_pra()
        self.create_mic(mic_num,mic_radius)

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
            "SRR " :self.SRR,
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


    def set_host_label(self,host_label):
        if host_label == None:
            item = self.listdata[0]
            host_label = item["label"]
            print(f"set {host_label} host")
            self.speech_host_label = host_label
        else:
            index = next((item for item in self.listdata if item["label"] == host_label), None)
            if index is not None:
                print(f"set {host_label} host")
            else:
                item = self.listdata[0]
                host_label = item["label"]
                print(f"set {host_label} host")
                self.speech_host_label = host_label
                

        
#generate room_type meeting_type rt60 room_size e_absorption max_order src_num
    def generate_room_pra(self,meeting_type):
        if self.rt60 == None:
            if len(self.listdata)<20:
                self.rt60 = round(random.uniform(0.2,0.8),2)
            else:
                self.rt60 = round(random.uniform(0.2,1),2)
                
        if self.room_size == None:
            if len(self.listdata)<20:
                self.room_size = [random.uniform(8,10), random.uniform(7,8), random.uniform(4,5)]
                self.room_type = "middle"
            else:
                self.room_size = [random.uniform(10,12), random.uniform(8,10), random.uniform(5,6)]
                self.room_type = "large"
 
        if self.meeting_type == None:
            if len(self.listdata)<20:
                self.meeting_type = meeting_type[random.randint(0,2)]
            else:
                self.meeting_type = random.choice([meeting_type[0],meeting_type[2]])

        print(self.room_size)
        print(self.meeting_type)
        #self.rt60,self.room_size = self.suggest_rt60_roomsize()
        self.e_absorption, self.max_order = pra.inverse_sabine(self.rt60,self.room_size)

        


#generate center radius width length
    def generate_src_pra(self):
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


    def create_room(self):
        self.room = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            materials=pra.Material(self.e_absorption),
            max_order=self.max_order,
            ray_tracing=True,
            air_absorption=True,
        )

    def create_mic(self,mic_num,mic_radius):
        
        if self.mic_num == None:
            self.mic_num = mic_num[random.randint(0,1)]
            #self.mic_num = 6
        if self.mic_radius == None:
            self.mic_radius = mic_radius[random.randint(0,len(mic_radius)-1)]
        mic_height =[round(random.uniform(0.8,1),2),round(random.uniform(self.room_size[2]-1,self.room_size[2]-self.d),2)]#desk,ceiling
        if self.array_num == None:
            self.array_num = 2
            self.SRR_circle2 = []


        #add circle mic array
        for i in range(self.array_num):
            if self.meeting_type == 'desk':
                self.mic_pos = [self.d+(i+1)*(self.width-self.d)/(self.array_num+1),self.room_size[1]/2]
                h=mic_height[0]#desk
            elif self.meeting_type == 'circle':

                self.mic_pos = [self.src_center[0]-self.max_radius+(i+1)*2*self.max_radius/(self.array_num+1),self.room_size[1]/2]
                h = mic_height[0]#desk

            elif self.meeting_type == 'speech':
                self.mic_pos = [self.src_center[0]-self.max_radius+(i+1)*2*self.max_radius/(self.array_num+1),self.room_size[1]/2]
                h = mic_height[1]#ceiling

            
            #self.mic_pos = [self.room_size[0]/2,self.room_size[1]/2,mic_height[random.randint(0,1)]]
        
            R = pra.circular_2D_array(self.mic_pos,self.mic_num,0,self.mic_radius)#circular_2D_array(center, M, phi0, radius)
            if self.mic_num == 6:
                new_column = np.array(self.mic_pos).reshape(-1, 1)  
                R = np.hstack((R, new_column))#6+1

            z_coords = np.full(R.shape[1], h)  
            R = np.vstack([R, z_coords])
            self.mic_loc.append(R)
            self.mic_center_loc.append(self.mic_pos)

            self.room.add_microphone_array(R)

        #add linear mic array
        h = round(random.uniform(1.6,1.8),2)
        d = 0.05
        self.linear_array_center = [self.room_size[0]-self.d,self.room_size[1]/2]
        self.mic_center_loc.append(self.linear_array_center)

        #fixed distance 
        linear_mic_pos = self.random_linear_array(self.linear_array_center)

        z_coords = np.full(linear_mic_pos.shape[1], h)  
        linear_mic_pos = np.vstack([linear_mic_pos, z_coords])
        #print(linear_mic_pos)
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
    
    def read_listfile2(self,list_path,filename):
        
        with open(os.path.join(list_path,filename), "r") as f:
            for line in f:
                line = line.strip() 
                if not line:  
                    continue     
                
                parts = line.split()

                if len(parts) >= 4:
                    self.vad_label.append((float(parts[0]),float(parts[1])))
                    index = next((item for item in self.listdata if item["label"] == parts[2]), None)
                    if index is not None:
                        if float(parts[0]) > index["end"]:
                            index["end"] = float(parts[1])
                            index["start_time"].append(float(parts[0]))
                            index["end_time"].append(float(parts[1]))
                            index["file_path"].append(parts[3])
                    else:
                        self.listdata.append({
                            "start_time": [float(parts[0])],  
                            "end_time": [float(parts[1])],    
                            "label": parts[2],              
                            "file_path": [parts[3]],
                            "end": float(parts[1]),
                        })

                    if float(parts[1]) > self.audio_len:
                        self.audio_len = float(parts[1])

        start = 0
        for item in self.vad_label:
            if item[0] >= start:
                self.vad_dur += item[1] - item[0]
            else:
                self.vad_dur += item[1] - start
            start = item[1]
        self.vad_dur = int(self.vad_dur)
        print(f"vad_dur : {self.vad_dur}")
        print(f"src_num: {len(self.listdata)}")


        

    def merge_audio(self,listdata):

        file_paths = listdata["file_path"]
        start_time = 0
        
        for index, path in enumerate(file_paths):
            audio_path = os.path.join(self.wav_path,path)
            audio, fs = sf.read(audio_path)
            if fs != self.fs:
                audio = self.resample(audio,fs,self.fs)
                fs = self.fs
            #print(index)
            if index == 0:
                #audio = np.pad(audio, (int(listdata["start_time"][index]*fs), 0), mode='constant')
                final_audio = audio
            else:
                if(listdata["start_time"][index]<start_time):
                    print("当前：",path,listdata["start_time"][index])
                    raise ValueError("发现无效的时间戳")
                start = listdata["start_time"][index]
                tmp = (start-start_time)*fs
                audio = np.pad(audio, (round(tmp), 0), mode='constant')
                final_audio = np.concatenate([final_audio, audio])
            
            start_time = listdata["end_time"][index]
        
        return final_audio
    

    def set_pos(self):
        if self.meeting_type == 'circle':
            return self.set_pos_circle_v1()
        elif self.meeting_type == 'desk':
            return self.set_pos_table_v2()
        elif self.meeting_type == 'speech':
            return self.set_pos_speech()

    def set_pos_circle_v1(self):
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

    def set_pos_table_v2(self):
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
    
    def set_pos_speech(self):
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
            #noise = (self.set_gain(torch.from_numpy(noise),-200)).numpy()

            DSC, Cs = gen_desired_spatial_coherence(pos_mics=pos_mics, fs=fs, noise_field='spherical', nfft=nfft)
            
            L = int(noise.shape[0] / Cs.shape[1])
            diffuse_noise.append(gen_diffuse_noise(noise=noise, L=L, Cs=Cs, nfft=nfft))
            

        return diffuse_noise

    def prepare_point_noise(noise_csv_path,typelist_path):

        target_categories = []
        with open(typelist_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    category = parts[0]
                    position = parts[1]

                    target_categories.append((category, position))

        df = pd.read_csv(noise_csv_path)

        category_files = defaultdict(list)
        for idx, row in df.iterrows():
            filename = row['filename']
            category = row['category']
            category_files[category].append(filename)
        

        return category_files,target_categories

    def resample(self,y, original_sample_rate, target_sample_rate: int = 16_000):
        return signal.resample(y, int(len(y) * target_sample_rate / original_sample_rate))

    def gen_point_noise(self,category_files,target_categories,noise_path,min_segments = 15,max_segments = None):

        point_noise = []
        
        self.room_for_noise = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            materials=pra.Material(self.e_absorption),
            max_order=self.max_order,
            ray_tracing=True,
            air_absorption=True,
        )

        for i,item in enumerate(self.mic_loc):
            self.room_for_noise.add_microphone_array(item)

        if max_segments == None:
            max_segments = int(self.audio_len/16)
        point = np.random.randint(min_segments, max_segments)

        intervals = np.random.exponential(15.0, size=point)
        interval = np.clip(intervals,8,None)
        time_cursor = 0.0
        noise_num = 0
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
            self.point_noise_time += dur
            if end > self.audio_len:
                print(f"add {noise_num} point noise")
                break  
            time_cursor = end

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
                
            if noise_list[0] == 'music':
                tmp = 5*self.fs
                self.room_for_noise.add_source(noise_pos, signal=noise[tmp:tmp+min(tmp,len(noise))], delay = start)
                
            else:
                self.room_for_noise.add_source(noise_pos, signal=noise[:min(5* self.fs,len(noise))], delay = start)
            noise_num+=1

        self.room_for_noise.simulate()
        array_num = 0
        for i in range(len(self.mic_loc)):
            array_signal = self.room_for_noise.mic_array.signals[array_num : array_num + len(self.mic_loc[i][1])]
            array_num += len(self.mic_loc[i][1])
            
            point_noise.append(array_signal)
        
        return point_noise

    def compute_DRR(self,h, fs, t_direct_ms=5):
        t_direct = int(fs * t_direct_ms / 1000)
        peak_idx = np.argmax(np.abs(h))
        direct_part = h[peak_idx : peak_idx + t_direct]
        reverb_part = np.concatenate([h[:peak_idx], h[peak_idx + t_direct:]])
        
        energy_direct = np.sum(direct_part ** 2)
        energy_reverb = np.sum(reverb_part ** 2)
        drr_db = 10 * np.log10(energy_direct / (energy_reverb + 1e-12))
        return drr_db
    
    def compute_SRR(self,s, h, fs, t_early_ms=50):
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
    


    def set_gain(self,signal,target_db,reference_pressure=20e-6):

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
        

    def set_speech_host_pos(self,listdata,split_size=100):

        src_height = round(random.uniform(1.6,1.8),2)


        start_pos = [random.uniform(self.room_size[0] - 3*self.d_wall,self.room_size[0] - 2*self.d_wall),random.uniform(self.room_size[1]/2+self.d_wall/2,self.room_size[1]-self.d_wall),src_height]
        end_pos = [random.uniform(self.room_size[0] - 3*self.d_wall,self.room_size[0] - 2*self.d_wall),random.uniform(self.d_wall,self.room_size[1]/2-self.d_wall/2),src_height]

        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        dist_total = norm(start_pos - end_pos)
        unit_vec = (start_pos - end_pos) / dist_total
        
        num_points = int(math.sqrt(len(listdata["start_time"])))
        print(f"len of moving point: {num_points}")

        alphas = np.linspace(0, 1, num_points)
        pos_src = np.array([start_pos + alpha * (end_pos - start_pos) for alpha in alphas])
        pos_src = [start_pos + alpha * (end_pos - start_pos) for alpha in alphas]

        host_DRR = []
        signal_gain = random.uniform(1,5)
        self.gains.append(signal_gain)

        file_paths = listdata["file_path"]

        avg = len(listdata["start_time"]) // num_points
        remainder = len(listdata["start_time"]) % num_points
        start_index = 0
        for i in range(num_points):
            chunks = avg + (1 if i <remainder else 0)
            
            for index in range(chunks):
                audio_path = os.path.join(self.wav_path,file_paths[start_index + index])
                audio, fs = sf.read(audio_path)
                if fs != self.fs:
                    audio = self.resample(audio,fs,self.fs)
                    fs = self.fs
                if index == 0:
                    final_audio = audio
                else:
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
            if self.is_compute_DRR:
                host_DRR.append(self.compute_DRR(final_audio,self.fs))
            start_index += chunks


        if self.is_compute_DRR:
            #print("host_DRR:",np.mean(host_DRR))    
            self.DRR.append(np.mean(host_DRR))
        


    def _add_noise(self,speech_sig, vad_duration, point_noise,diffuse_noise, SNR_point,SNR_diffuse):
        """add noise to the audio.
        :param speech_sig: The input audio signal (Tensor).
        :param vad_duration: The length of the human voice (int).
        :param noise_sig: The input noise signal (Tensor).
        :param snr: the SNR you want to add (int).
        :returns: noisy speech sig with specific snr.
        """

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
                if target_length > noise.shape[0]:
                    
                    repeat_times = int(np.ceil(target_length / noise.shape[0]))
                    repeated_noise = noise.repeat(repeat_times+1,1)
                    return repeated_noise[:target_length,:]
                else:
                    
                    return noise[ :target_length,:]
                
            point_noise_sig = adjust_noise_length(point_update, speech_sig.shape[0])

            diffuse_noise_sig = adjust_noise_length(diffuse_update, speech_sig.shape[0])

            return speech_sig + diffuse_noise_sig + point_noise_sig
        
        else:
            return speech_sig






    