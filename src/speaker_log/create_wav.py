# -*- coding: utf-8 -*-
# @Date     : 2025/05/10
# @Author   : Getsum
# @File     : create_wav.py
# @Github   : https://github.com/getsum-zero

from argparse import ArgumentParser
import os
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('log.txt')),
    ]
)

parser = ArgumentParser(add_help=True)
parser.add_argument('--logging_file', type=str)
parser.add_argument('--output_dir', type=str)   
parser.add_argument('--sample_rate', type=int, default=16000)
args = parser.parse_args()

if __name__ == '__main__':
    max_lens = 0
    with open(args.logging_file, 'r') as f:
        for line in f:
            keys = line.strip().split()
            max_lens = max(max_lens, float(keys[1]))
    max_lens = int(max_lens * args.sample_rate)
    wavs = np.zeros(max_lens, dtype=np.float32)
    with open(args.logging_file, 'r') as f:
        for line in f:
            keys = line.strip().split()
            st, ed = int(float(keys[0]) * args.sample_rate), int(float(keys[1]) * args.sample_rate)
            waveform, sr = sf.read(keys[3], dtype='float32')
            assert sr == args.sample_rate, f"Sample rate mismatch: {sr} != {args.sample_rate}"
            waveform = waveform[:, 0] if len(waveform.shape) > 1 else waveform

            cutting_start_time = int(float(keys[4]) * args.sample_rate)
            duration = ed - st
            lens = min(len(waveform)-cutting_start_time, duration)
            try:
                waveform = waveform /  np.max(np.abs(waveform)) * 0.9
            except:
                pass
            wavs[st:st+lens] = wavs[st:st+lens] + waveform[cutting_start_time:cutting_start_time+lens]
    
    wavs = wavs / np.max(np.abs(wavs))
    wavs = wavs * 0.9

    basename = os.path.basename(args.logging_file).split('.')[0]
    sf.write(os.path.join(args.output_dir, f'{basename}.wav'), wavs, args.sample_rate)


