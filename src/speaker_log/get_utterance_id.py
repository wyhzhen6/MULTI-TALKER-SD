# -*- coding: utf-8 -*-
# @Date     : 2025/05/09
# @Author   : Getsum
# @File     : get_utterance_id.py
# @Github   : https://github.com/getsum-zero

from argparse import ArgumentParser
import os
import json
import logging
import yaml
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
parser.add_argument('--speaker_id', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--output_dir', type=str)   
args = parser.parse_args()


def get_utterance_number(ranges):
    try:
        ranges, distru = ranges.split(':')
        mins, maxs = ranges.split('-')
        if distru == 'unif':
            num = np.random.randint(int(mins), int(maxs) + 1)
        else:
            return NotImplementedError
        return num
    except ValueError:
        logging.error(f"[utterance] should be in the format 'start(int)-end(int)', but got {ranges}")
        raise ValueError(f"[utterance] should be in the format 'start(int)-end(int)', but got {ranges}")

def read(speaker_list):
    spk2utt = {}
    for speaker in speaker_list:
        assert os.path.isfile(speaker), f"speaker should be a file, but got {speaker}"
        utts = json.load(open(speaker, 'r'))
        speaker_id = os.path.basename(speaker).split('.')[0]
        spk2utt[speaker_id] = utts
    return spk2utt


def get_main_speaker(spk2utt, speaker_num, main_speaker_num, utterance_num, main_utterance_num):
    '''
        Since the number of utts of the randomly selected main speaker may be less than the predetermined number, 
        multiple attempts are required
    '''
    count = 0
    while count < speaker_num:
        main_speaker = set(np.random.choice(list(spk2utt.keys()), main_speaker_num, replace=False))
        other_speaker = set(spk2utt.keys()) - main_speaker
    
        main_utt_list = []
        other_utt_list = []
        for speaker in main_speaker:
            main_utt_list += spk2utt[speaker]
        if main_utterance_num > len(main_utt_list):  # The number of utts of the selected main speaker is less than the preset
            count = count + 1
            continue
        else:
            main_utt_list = np.random.choice(main_utt_list, main_utterance_num, replace=False).tolist()
            for speaker in other_speaker:
                other_utt_list += spk2utt[speaker]
            other_utterance_num = min(utterance_num - main_utterance_num, len(other_utt_list))
            other_utt_list = np.random.choice(other_utt_list, other_utterance_num, replace=False).tolist()
            return main_utt_list, other_utt_list

    if count == speaker_num:
        logging.info(f"main speaker number {main_speaker_num} is too large, re-edit the preset value")
        main_speaker = set(np.random.choice(list(spk2utt.keys()), main_speaker_num, replace=False))
        other_speaker = set(spk2utt.keys()) - main_speaker
        main_utt_list = []
        other_utt_list = []
        for speaker in main_speaker:
            main_utt_list += spk2utt[speaker]
        other_utterance_num = int(len(main_utt_list) * (utterance_num - main_utterance_num) / main_utterance_num)
        for speaker in other_speaker:
            other_utt_list += spk2utt[speaker]
        other_utterance_num = min(other_utterance_num, len(other_utt_list))
        other_utt_list = np.random.choice(other_utt_list, other_utterance_num, replace=False).tolist()
        return main_utt_list, other_utt_list




def choose_utt(spk2utt, utterance_num, conf, meeting_type):
    utt_list = []

    # ==========================================================================
    # New meeting types or generation strategies can be added here
    # ==========================================================================
    if meeting_type == 'presentation':
        speaker_num = len(spk2utt.keys())
        # get main speaker number
        main_speaker_num, main_utterance_num = conf['main_speaker'].split(':')
        main_speaker_l, main_speaker_r = main_speaker_num.split('-')
        main_utterance_l, main_utterance_r = main_utterance_num.split('-')
        main_speaker_num = int((np.random.rand() * (float(main_speaker_r) - float(main_speaker_l)) + float(main_speaker_l)) * speaker_num) 
        main_speaker_num = 1 if main_speaker_num == 0 else main_speaker_num
        main_utterance_num = int(np.random.rand() * (float(main_utterance_r) - float(main_utterance_l)) + float(main_utterance_l) * utterance_num)
        main_utt_list, other_utt_list = get_main_speaker(spk2utt, speaker_num, main_speaker_num, utterance_num, main_utterance_num)
        # main_utt_list = [{'path': utt['path'], 'duration': utt['duration'], 'speaker_id': utt['speaker_id'], 'main_speaker': 1} for utt in main_utt_list]
        # other_utt_list = [{'path': utt['path'], 'duration': utt['duration'], 'speaker_id': utt['speaker_id'], 'main_speaker': 0} for utt in other_utt_list]
        
        main_utt_list = [ {**utt, 'main_speaker': 1} for utt in main_utt_list ]
        other_utt_list = [ {**utt, 'main_speaker': 0} for utt in other_utt_list ]

        split_num = len(main_utt_list)
        utt_list = main_utt_list + other_utt_list
    
    elif meeting_type == 'interview':
        speaker_num = len(spk2utt.keys())
        interviewer_ratio_l, interviewer_ratio_r = conf['interviewer_ratio'].split('-')
        interviewer_utterance_num = int(np.random.rand() * (float(interviewer_ratio_r) - float(interviewer_ratio_l)) + float(interviewer_ratio_l) * utterance_num)
        main_utt_list, other_utt_list = get_main_speaker(spk2utt, speaker_num, 1, utterance_num, interviewer_utterance_num)
        # main_utt_list = [{'path': utt['path'], 'duration': utt['duration'], 'speaker_id': utt['speaker_id'], 'main_speaker': 1} for utt in main_utt_list]
        # other_utt_list = [{'path': utt['path'], 'duration': utt['duration'], 'speaker_id': utt['speaker_id'], 'main_speaker': 0} for utt in other_utt_list]
        main_utt_list = [ {**utt, 'main_speaker': 1} for utt in main_utt_list ]
        other_utt_list = [ {**utt, 'main_speaker': 0} for utt in other_utt_list ]

        split_num = len(main_utt_list)
        utt_list = main_utt_list + other_utt_list
        


    elif meeting_type == 'discussion':
        for _ , values in spk2utt.items():
            utt_list += values
        if len(utt_list) < utterance_num:
            logging.warning(f"meeting type {meeting_type} has only {len(utt_list)} utterances, but need {utterance_num}")
            utterance_num = len(utt_list)
        utt_list = np.random.choice(utt_list, utterance_num, replace=False).tolist()
        #utt_list = [{'path': utt['path'], 'duration': utt['duration'], 'speaker_id': utt['speaker_id']} for utt in utt_list]
        split_num = -1
    else:
        logging.warning(f"meeting type {meeting_type} is not supported, please check the config file")

    duration = np.sum([utt['duration'] for utt in utt_list])
    return utt_list, duration, split_num
    


if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)
    config = yaml.safe_load(open(args.config, 'r'))

    # seed
    np.random.seed(config['seed'])
    
    # meeting type
    conf = config['meeting']
    meeting_types = list(conf.keys())
    meeting_sizes = len(meeting_types)
    meeting_weights = [conf[meeting]['type_weight'] for meeting in meeting_types]
    assert np.sum(meeting_weights) > 0, f"meeting weights should be greater than 0, but got {meeting_weights}"
    meeting_weights = np.array(meeting_weights) / np.sum(meeting_weights)
    logging.info(f"meeting_types: {meeting_types}")
    logging.info(f"meeting_weights: {meeting_weights}")

    utterance_id = os.path.join(args.output_dir, 'utterance_id.list')
    tot_num = 0
    with open(args.speaker_id, 'r') as f:
        tot_num = np.sum([1 for _ in f])

    pbar = tqdm(total=tot_num, desc='Processing', unit='file')
    with open(args.speaker_id, 'r') as f, open(utterance_id, 'w') as f_out:
        for line in f:
            # get utterance number
            utterance_num = get_utterance_number(config['utterance'])

            speakers = line.strip().split()
            spk2utt = read(speakers)
            
            meeting_type = meeting_types[np.random.choice(meeting_sizes, p=meeting_weights)]
            utt_list, duration, split_num = choose_utt(spk2utt, utterance_num, config['meeting'][meeting_type], meeting_type)
            # print(f"utt_list: {utt_list}")

            data = {
                'meeting_type': meeting_type,
                'utterance_num': utterance_num,
                'duration': duration,
                'split_num': split_num,
                'utt_list': utt_list,
            }
            json.dump(data, f_out, ensure_ascii=False)
            f_out.write('\n')
            pbar.update(1)
    pbar.close()
    logging.info(f"utterance_id list saved to {utterance_id}")




    
    
