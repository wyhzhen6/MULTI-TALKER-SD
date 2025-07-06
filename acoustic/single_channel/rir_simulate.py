import argparse
import os
import time
import soundfile as sf
import json
import numpy as np
import math
import random
import pandas as pd
import torch


import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra

from set_rir_room import rir_room


def simulate(room,simulate_config,filepath):
    start_time = time.time()
    list_path             = simulate_config.get('list_path', )  
    wav_path              = simulate_config.get('wav_path', )   
    output_path           = simulate_config.get('output_path', )   
    host_label            = simulate_config.get('host_label', )


    if room.meeting_type == "speech":
        room.set_host_label(host_label)

    audio = []
    #set src pos
    for index, item in enumerate(room.listdata):
        if item["label"] != room.speech_host_label:
            src_pos = room.set_pos()
            signals = room.merge_audio(item)

            signal_gain = random.uniform(1,5)
            #print(f"signal_gain: {signal_gain}")
            room.gains.append(signal_gain)
            signals = room.set_gain(torch.from_numpy(signals),signal_gain).numpy()
            if room.is_compute_SRR:
                audio.append(signals)
            
            room.room.add_source(src_pos, signal=signals, delay=item["start_time"][0])
            if room.is_compute_DRR:
                #compute DRR
                room.DRR.append(room.compute_DRR(signals,room.fs))
    
    if room.meeting_type == "speech":
        item = next((item for item in room.listdata if item["label"] == room.speech_host_label), None)
        room.set_speech_host_pos(item)

    room.room.simulate()
    print("finish simulate")
    mid_time =  time.time()
    print(f"simulate _time: {mid_time - start_time}")
    #compute SRR
    if room.is_compute_SRR:
        for i in range(len(audio)):
            rir1 = room.room.rir[0][i]
            room.SRR.append(room.compute_SRR(audio[i],rir1,room.fs))

        if room.meeting_type == "speech":
            
            item = next((item for item in room.listdata if item["label"] == room.speech_host_label), None)
            host_srr = []
            for index2,audio_host in enumerate(room.host_audio):

                rir1 = room.room.rir[0][index2+len(audio)] 
                
                host_srr.append(room.compute_SRR(audio_host,rir1,room.fs))

            
            room.SRR.append(np.mean(host_srr))    
        
        room.SRR = np.mean(room.SRR)

    if room.is_compute_DRR:
        room.drr = np.mean(room.DRR)



    #room.room.simulate()
    filename,tmp = os.path.splitext(filepath)
    

    #save clean audio

    clean_signal = room.room.mic_array.signals
    if not os.path.exists(os.path.join(output_path,filename)):
        os.makedirs(os.path.join(output_path,filename))
    output_path_clean = os.path.join(output_path,filename, f"clean_signal.wav")
    sf.write(
        output_path_clean,  
        clean_signal.T,         
        room.fs,                   
        subtype="PCM_16"     
    )
    print(f"已保存干净语音: {output_path_clean}")

    

            
def add_noise(room,category_files,target_categories,simulate_config,filepath):
    start_time = time.time()
    point_noise_path      = simulate_config.get('point_noise_path', )
    diffuse_noise_path    = simulate_config.get('diffuse_noise_path', ) 
    output_path           = simulate_config.get('output_path', ) 
    SNR_point             = simulate_config.get('SNR_point', )
    SNR_point_arr         = simulate_config.get('SNR_point_arr', )
    SNR_diffuse           = simulate_config.get('SNR_diffuse', )    
    SNR_diffuse_arr       = simulate_config.get('SNR_diffuse_arr', )      
    filename,tmp = os.path.splitext(filepath)
    print("add noise")
    #gen point_noise
    point_noise = room.gen_point_noise(category_files,target_categories,point_noise_path)
    diffuse_file = random.choice(os.listdir(diffuse_noise_path))
    diffuse_noise, fs = sf.read(os.path.join(diffuse_noise_path,diffuse_file))
    
    if fs != room.fs:
        noise = room.resample(noise,fs,room.fs)
        fs = room.fs
    
    diffuse_noise = diffuse_noise[:,np.newaxis]
    
    if SNR_point == None:
        SNR_point = random.randint(SNR_point_arr[0],SNR_point_arr[1])
    room.SNR_point = SNR_point

    if SNR_diffuse == None:
        SNR_diffuse = random.randint(SNR_diffuse_arr[0],SNR_diffuse_arr[1])
    room.SNR_diffuse = SNR_diffuse


    clean_signal = room.room.mic_array.signals

    final_signal = room._add_noise(speech_sig = torch.from_numpy(clean_signal.T),vad_duration = room.vad_dur,point_noise = torch.from_numpy(point_noise.T),diffuse_noise = torch.from_numpy(diffuse_noise), SNR_point = room.SNR_point,SNR_diffuse = room.SNR_diffuse)
    
    final_signal = final_signal.numpy()
    if not os.path.exists(os.path.join(output_path,filename)):
        os.makedirs(os.path.join(output_path,filename))
    output_path_final = os.path.join(output_path,filename, f"final_signal.wav")
    sf.write(
        output_path_final,  
        final_signal,         
        room.fs,                   
        subtype="PCM_16"      
    )
    print(f"已保存融合噪声语音: {output_path_final}")

    
    info_path = f"{filename}_single_info.json"

    with open(os.path.join(output_path,filename,info_path), "w", encoding="utf-8") as file:
        file.write(room.to_json())

    end_time = time.time()
    print(f"噪声添加时间：{end_time - start_time:.6f} 秒")

def rir_simulate(simulate_config):
    list_path             = simulate_config.get('list_path', )      
    point_noise_csv       = simulate_config.get('point_noise_csv', )
    point_noise_type      = simulate_config.get('point_noise_type', )

    category_files,target_categories = rir_room.prepare_point_noise(point_noise_csv,point_noise_type)


    for filepath in os.listdir(list_path):
        if filepath.endswith('.list'):
            start_time= time.time()
            list_path             = simulate_config.get('list_path', )  
            wav_path              = simulate_config.get('wav_path', )   
            fs                    = simulate_config.get('fs', 16000)
            #create room("small","medium","large")
            room = rir_room(
                wav_path = wav_path,
                list_path = list_path,
                filepath = filepath,
                fs = fs,
                config = simulate_config,
                rt60 = simulate_config['rt60'],
                room_size = simulate_config['room_size'],
                room_type = simulate_config['room_type'],
                meeting_type = simulate_config['meeting_type'],        
                mic_pos = simulate_config['mic_pos'],
                is_compute_DRR = simulate_config['is_compute_DRR'],
                is_compute_SRR = simulate_config['is_compute_SRR']
            )
            simulate(
                room = room,
                simulate_config = simulate_config,
                filepath = filepath
                )
            add_noise(
                room = room,
                category_files = category_files,
                target_categories = target_categories,
                simulate_config = simulate_config,
                filepath = filepath   
                )
            end_time = time.time()
            print(f"程序运行时间{end_time - start_time:.6f} 秒")
            print("finish")
            



if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Make datasets for rir')
    parser.add_argument('-c', '--config', type=str, default='/data/zjj/rir/rir_single/script/rir_simulate.json',
                        help='JSON file for configuration')

    args=parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    simulate_config = config["simulate_config"]

    rir_simulate(simulate_config)
