# -*- coding: utf-8 -*-
# @Date     : 2025/05/05
# @Author   : Getsum
# @File     : aishell-1.py
# @Github   : https://github.com/getsum-zero

# @Description:
# 1. This script is only applicable to the aishell-1 dataset

from argparse import ArgumentParser
import os
import glob
import json
import soundfile as sf
import logging

parser = ArgumentParser(add_help=True)
parser.add_argument('--aishell_1_dir', type=str)
parser.add_argument('--output_dir', type=str)   # save in Mandarin subset
parser.add_argument('--file_type', type=str, default='.wav')
parser.add_argument('--transcript', type=str, default=None)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('log.txt')),
    ]
)
logger = logging.getLogger(__name__)
logger.propagate = False

def get_gerder(data_path):
    gerder_path = os.path.join(data_path, "resource_aishell/speaker.info")
    if os.path.exists(gerder_path):
        with open(gerder_path, 'r') as f:
            lines = f.readlines()

        gerder_dict = {}
        for line in lines:
            line = line.strip().split(' ')
            gerder_dict[line[0].strip()] = line[1].strip()
        return gerder_dict
    else:
        print(f"File {gerder_path} not found.")
        raise Exception
    

def get_utterance(data_path, transcripts=None):
    wavs_generator = glob.iglob(os.path.join(data_path, f'**/*{args.file_type}'), recursive=True)
    spk2utt = {}
    for wav in wavs_generator:
        wav_name = os.path.basename(wav)

        # such as BAC009S0002W0122
        # BAC009-S0002-W0122
        speaker_id = wav_name.split('.')[0][7:11]

        if speaker_id not in spk2utt:
            spk2utt[speaker_id] = []

        key = wav_name.split('.')[0]

        transcript = None
        if transcripts is not None:
            try:
                transcript = transcripts[key]
            except KeyError:
                logging.warning(f"key {key} not in transcript, skip it")
                

        spk2utt[speaker_id].append({
            "key": f'aishell_1_{key}',
            "speaker_id": f'aishell_1_{speaker_id}',
            "path": os.path.abspath(wav),
            "duration": sf.info(wav).duration,
            # "subset": '/'.join(wav.split('/')[-4:-2]),
            "Transcription": transcript,
        })

    return spk2utt

def get_transcript(transcript):
    text = {}
    with open(transcript, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            key = line[0]
            text[key] = ' '.join(line[1:])
    return text


if __name__ == '__main__':

    assert(args.output_dir.endswith('Chinese'))
    output_dir  = os.path.join(args.output_dir, "aishell-1")
    os.makedirs(output_dir, exist_ok=True)
    
    gender_mapping = get_gerder(args.aishell_1_dir)
    transcripts = None
    if args.transcript is not None:
        transcripts = get_transcript(args.transcript)
    utterance_metadata = get_utterance(args.aishell_1_dir, transcripts)

    utt_meta = os.path.join(output_dir, "utterance")
    os.makedirs(utt_meta, exist_ok=True)
    speaker_metadata_male = [] 
    speaker_metadata_female = [] 


    for key, value in utterance_metadata.items():

        try:
            gender = gender_mapping[key]
        except KeyError:
            logging.warning(f"speaker {key} not in speaker.info, skip it")
            continue

        try:
            assert gender in ['M', 'F']
        except AssertionError:
             logging.warning(f"Unkown {key}'s gender in speaker.info, skip it")
        

        utt_file = os.path.join(utt_meta, f"aishell_1_{key}.json")
        with open(utt_file, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=4, ensure_ascii=False)
        

        speaker_utt = {
                "speaker_id":  f'aishell_1_{key}',
                "gender": "male" if gender == 'M' else "female",
                'utterance_metadata': os.path.abspath(utt_file),
                "language": "chinese",
                "source": "aishell-1"
            }
        if gender == 'M':
            speaker_metadata_male.append(speaker_utt)
        else:
            speaker_metadata_female.append(speaker_utt)
        

    with open(os.path.join(output_dir, "speakers_male.json"), 'w') as f:
            json.dump(speaker_metadata_male, f, indent=4)
    with open(os.path.join(output_dir, "speakers_female.json"), 'w') as f:
            json.dump(speaker_metadata_female, f, indent=4)
        
                
                