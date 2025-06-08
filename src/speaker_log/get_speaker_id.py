# -*- coding: utf-8 -*-
# @Date     : 2025/05/05
# @Author   : Getsum
# @File     : get_speaker_id.py
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
parser.add_argument('--metadata_dir', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--output_dir', type=str)   
args = parser.parse_args()


def normalize(prob, name):
    if np.sum(prob) != 1.0:
        if np.sum(prob) == 0.0:
            logging.warning(f"[speaker][{name}] probability should not be all zero, default to uniform distribution")
            prob = [1.0 / len(prob) for _ in range(len(prob))]
        else:
            logging.warning(f"[speaker][{name}] probability should sum to 1.0, but got {np.sum(prob)}, normalizing")
            prob = [p / np.sum(prob) for p in prob]
    return prob

def get_speaker_number(number_list):
    number_range = []
    number_prob = []
    for item in number_list:
        try:
            assert isinstance(item, str), f"item should be str, but got {type(item)}"
            ranges, pro = item.split(':')
            st, ed = ranges.split('-') 
            for i in range(int(st), int(ed) + 1):
                number_range.append(i)
                number_prob.append(float(pro) / (int(ed) - int(st) + 1))
        except ValueError:
            logging.error(f"[speaker][number] should be in the format 'start-end:prob', but got {item}")
            raise ValueError(f"[speaker][number] should be in the format 'start-end:prob', but got {item}")
    
    number_prob = normalize(number_prob, 'number')
    logging.info(f"number_range: {number_range}")
    logging.info(f"number_prob: {number_prob}")
    return number_range, number_prob


def get_speaker_gender(gender_list):
    gender_choice, gender_prob = [], []
    for item in gender_list:
        try:
            assert isinstance(item, str), f"item should be str, but got {type(item)}"
            male_ratio, prob = item.split(':')
            gender_choice.append(float(male_ratio))
            gender_prob.append(float(prob))
        except ValueError:
            logging.error(f"[speaker][gender] should be in the format 'male_ratio:prob', but got {item}")
            raise ValueError(f"[speaker][gender] should be in the format 'male_ratio:prob', but got {item}")
    
    gender_prob = normalize(gender_prob, 'gender')
    logging.info(f"gender_choice: {gender_choice}")
    logging.info(f"gender_prob: {gender_prob}")
    return gender_choice, gender_prob


def get_speaker_lang(lang_list):
    lang_choice, lang_prob = [], []
    for item in lang_list:
        try:
            assert isinstance(item, str), f"item should be str, but got {type(item)}"
            eng_ratio, prob = item.split(':')
            if eng_ratio == 'unif':
                eng_ratio = np.random.rand()
            lang_choice.append(float(eng_ratio))
            lang_prob.append(float(prob))
        except ValueError:
            logging.error(f"[speaker][language] should be in the format 'english_ratio:prob', but got {item}")
            raise ValueError(f"[speaker][language] should be in the format 'english_ratio:prob', but got {item}")
    
    lang_prob = normalize(lang_prob, 'language')
    logging.info(f"lang_choice: {lang_choice}")
    logging.info(f"lang_prob: {lang_prob}")
    return lang_choice, lang_prob


def get_metadata(metadata_dir):
    assert os.path.isdir(metadata_dir), f"{metadata_dir} is not a directory"

    subfolders = [f.path for f in os.scandir(metadata_dir) if f.is_dir()]
    logging.info(f"Datasets in {metadata_dir}: {subfolders}")

    male_list = []
    female_list = []
    for subfolder in subfolders:
        male = os.path.join(subfolder, 'speakers_male.json')
        female = os.path.join(subfolder, 'speakers_female.json')
        assert os.path.isfile(male) and os.path.isfile(female), f"{male} or {female} is not a file"
        with open(male, 'r', encoding='utf-8') as f:
            male_list.extend(json.load(f))
        with open(female, 'r', encoding='utf-8') as f:
            female_list.extend(json.load(f))
        logging.info('Load metadata from %s', subfolder)

    return [male_list, female_list]


def force_align(number, number_list):
    number_list_int = number_list.astype(int)
    if np.sum(number_list_int) != number:
        dif = number - np.sum(number_list_int)
        different = np.abs(number_list - number_list_int)
        while dif != 0:
            index = np.argmax(different)
            number_list_int[index] += 1
            different[index] = 0
            dif -= 1
    return number_list_int.astype(int)


if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)
    config = yaml.safe_load(open(args.config, 'r'))

    # setting
    np.random.seed(config['seed'])
    number_range, number_prob = get_speaker_number(config['speaker']['number'])
    gender_choice, gender_prob = get_speaker_gender(config['speaker']['gender'])
    lang_choice, lang_prob = get_speaker_lang(config['speaker']['language'])

    # get metadata  :  
    # [ 
    #   [ {speaker_id, utt_json, `male`, `english`}, ...] ,  
    #   [ {speaker_id, utt_json, `female`, `english`}, ...] ,
    #   [ {speaker_id, utt_json, `male`, `chinese`}, ...] ,  
    #   [ {speaker_id, utt_json, `female`, `chinese`}, ...] 
    # ]
    english = get_metadata(os.path.join(args.metadata_dir, 'English'))  # order by male-female
    chinese = get_metadata(os.path.join(args.metadata_dir, 'Chinese'))  
    metadata = english + chinese
    metadata_cnt = []
    for i in range(len(metadata)):
        metadata_cnt.append(np.ones(len(metadata[i])).astype(np.int32) * int(config['max_choose']))
        print(f"metadata[{i}] length: {len(metadata[i])}")
        logging.info(f"metadata[{i}] length: {len(metadata[i])}")




    speaker_id_json = os.path.join(args.output_dir, 'speaker_id.list')
    with open(speaker_id_json, 'w', encoding='utf-8') as f:

        for i in tqdm(range(config["iteration"])):
            speaker_id = []

            number = np.random.choice(number_range, p=number_prob)
            male_ratio = np.random.choice(gender_choice, p=gender_prob)
            eng_ratio = np.random.choice(lang_choice, p=lang_prob)
            # print(f"number: {number}, gender_ratio: {male_ratio}, lang_ratio: {eng_ratio}")

            number_list = number * np.array([\
                        male_ratio*eng_ratio, 
                        (1-male_ratio)*eng_ratio,
                        male_ratio*(1-eng_ratio),
                        (1-male_ratio)*(1-eng_ratio)
            ])
            
            # if rounded directly, it will easily be reset to zero 
            # number_list = np.round(number_list).astype(int)  # it is rounded directly and not aligned with the sampled `number`
            # number = np.sum(number_list)
            number_list = force_align(number, number_list)
            
            # check available speakers
            available_speakers = np.array([np.sum(cnt>0) for cnt in metadata_cnt])
            if np.sum(available_speakers >= number_list) < len(metadata_cnt):
                if np.sum(available_speakers) < config['stop_available']:
                    logging.info(f"Not enough speakers available, stop sampling!!")
                    break
                else:
                    continue
            
            # randomly choose speakers
            chosen = []
            for id, (num, subset, cnt) in enumerate(zip(number_list, metadata, metadata_cnt)):
                if num == 0:
                    continue
                prob = cnt / np.sum(cnt)
                random_sample = np.random.choice(len(subset), num, p=prob, replace=False)

                chosen.extend([subset[i]['utterance_metadata'] for i in random_sample])
                metadata_cnt[id][random_sample] -= 1
            
            assert len(chosen) == number, f"number of chosen speakers {len(chosen)} not equal to {number}"
            f.write(' '.join(chosen) + '\n')
                   
            


                
            
            
        
    
