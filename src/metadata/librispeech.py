# -*- coding: utf-8 -*-
# @Date     : 2025/05/05
# @Author   : Getsum
# @File     : librispeech.py
# @Github   : https://github.com/getsum-zero

# @Description:
# 1. This script is only applicable to the librispeech dataset

from argparse import ArgumentParser
import os
import glob
import json
import soundfile as sf
import logging

parser = ArgumentParser(add_help=True)
parser.add_argument('--librispeech_dir', type=str)
parser.add_argument('--output_dir', type=str)   # save in English subset
parser.add_argument('--file_type', type=str, default='.flac')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_gerder(data_path):
    gerder_path = os.path.join(data_path, "SPEAKERS.TXT")
    if os.path.exists(gerder_path):
        with open(gerder_path, 'r') as f:
            lines = f.readlines()

        gerder_dict = {}
        for line in lines:
            if line.startswith(";"):  continue
            line = line.strip().split('|')
            gerder_dict[line[0].strip()] = line[1].strip()
        return gerder_dict
    else:
        print(f"File {gerder_path} not found.")
        raise Exception
    

def get_utterance(data_path, transcription_mapping):
    wavs_generator = glob.iglob(os.path.join(data_path, f'**/*{args.file_type}'), recursive=True)
    spk2utt = {}
    for wav in wavs_generator:
        wav_name = os.path.basename(wav)
        speaker_id = wav_name.split('-')[0]
        if speaker_id not in spk2utt:
            spk2utt[speaker_id] = []

        key = wav_name.split('.')[0]
        text = transcription_mapping.get(key, None)
        spk2utt[speaker_id].append({
            "key": f'libri_{key}',
            "speaker_id": f'libri_{speaker_id}',
            "path": os.path.abspath(wav),
            "duration": sf.info(wav).duration,
            # "subset": '/'.join(wav.split('/')[-5:-3]), #wav.split('/')[-4],
            "Transcription": text,
        })
    return spk2utt

def get_transcription(data_path):
    txt_generator = glob.iglob(os.path.join(data_path, f'**/*.trans.txt'), recursive=True)
    trans_dict = {}
    for txt in txt_generator:
        with open(txt, 'r') as f:
            for line in f:
                keys = line.strip().split()
                trans_dict[keys[0]] = ' '.join(keys[1:]).strip()
    return trans_dict

if __name__ == '__main__':

    assert(args.output_dir.endswith('English'))
    output_dir  = os.path.join(args.output_dir, "librispeech")
    os.makedirs(output_dir, exist_ok=True)
    
    gender_mapping = get_gerder(args.librispeech_dir)
    transcription_mapping = get_transcription(args.librispeech_dir)
    utterance_metadata = get_utterance(args.librispeech_dir, transcription_mapping)

    utt_meta = os.path.join(output_dir, "utterance")
    os.makedirs(utt_meta, exist_ok=True)
    speaker_metadata_male = [] 
    speaker_metadata_female = [] 


    for key, value in utterance_metadata.items():

        try:
            gender = gender_mapping[key]
        except KeyError:
            logging.warning(f"speaker {key} not in SPEAKERS.TXT, skip it")
            continue

        try:
            assert gender in ['M', 'F']
        except AssertionError:
             logging.warning(f"Unkown {key}'s gender in SPEAKERS.TXT, skip it")
        

        utt_file = os.path.join(utt_meta, f"libri_{key}.json")
        with open(utt_file, 'w') as f:
            json.dump(value, f, indent=4)
        

        speaker_utt = {
                "speaker_id":  f'libri_{key}',
                "gender": "male" if gender == 'M' else "female",
                'utterance_metadata': os.path.abspath(utt_file),
                "language": "english",
                "source": "librispeech"
            }
        if gender == 'M':
            speaker_metadata_male.append(speaker_utt)
        else:
            speaker_metadata_female.append(speaker_utt)
        

    with open(os.path.join(output_dir, "speakers_male.json"), 'w') as f:
            json.dump(speaker_metadata_male, f, indent=4)
    with open(os.path.join(output_dir, "speakers_female.json"), 'w') as f:
            json.dump(speaker_metadata_female, f, indent=4)
        
                
                