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
from multiprocessing import Process, Manager
from multiprocessing import Value


import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra

from rir_room import singlechannel_rir_room
import yaml
from tqdm import tqdm


def process_chunk(
        chunk_lines, 
        simulate_config, 
        output_dir, 
        counter, 
        worker_id,
        point_noise_path,
        diffuse_noise_path
    ):
    
    for file in chunk_lines:
        room = singlechannel_rir_room(
            filepath = file,
            simulate_config = simulate_config
        )
        room.simulate(output_dir)

        room.add_noise(
            output_dir=output_dir,
            point_noise_path=point_noise_path,
            diffuse_noise_path=diffuse_noise_path
        )
        with counter.get_lock():
            counter.value += 1

def rir_simulate(simulate_config, args):
   
    num_workers = args.num_workers if args.num_workers > 0 else 1
    print(f"Using {num_workers} workers")
    
    lines = [f for f in os.listdir(args.logging_list) if f.endswith('.list')]
    lines = [os.path.join(args.logging_list, line) for line in lines ]
    chunk_size = (len(lines) + num_workers - 1) // num_workers
    chunks = [lines[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    
    manager = Manager()
    counter = Value('i', 0)
    
    
    processes = []
    for i in range(num_workers):
        p = Process(
            target=process_chunk,
            args=(chunks[i], 
                  simulate_config, 
                  args.output_dir, 
                  counter, 
                  i,
                  args.point_noise_path,
                  args.diffuse_noise_path
                )
        )
        p.start()
        processes.append(p)
    
    with tqdm(total=len(lines)) as pbar:
        prev = 0
        while any(p.is_alive() for p in processes):
            with counter.get_lock():
                now = counter.value
            pbar.update(now - prev)
            prev = now
            time.sleep(0.5)

        with counter.get_lock():
            pbar.update(counter.value - prev)

    for p in processes:
        p.join()


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Make datasets for rir')
    
    
    
    parser.add_argument('-c', '--config', type=str, help='configuration')
    parser.add_argument('--channels',type=str,choices=['mono', 'multi'],default='mono',help='Choose to generate single channel (mono) or multi channel (multi)')
    parser.add_argument('--logging_list', type=str, help='configuration')
    parser.add_argument('--point_noise_path', type=str, help='')
    parser.add_argument('--diffuse_noise_path', type=str, help='')
    parser.add_argument('--output_dir', type=str, help='')
    parser.add_argument("--num_workers", type=int, default=4, help="number of parallel worker processes (default=4)")

    args=parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, 'reverb'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'noisy'), exist_ok=True)
    
    config = yaml.safe_load(open(args.config, 'r'))
    simulate_config = config["simulate_config"]
    np.random.seed(config['seed']) # seed
    
    rir_simulate(simulate_config, args)
