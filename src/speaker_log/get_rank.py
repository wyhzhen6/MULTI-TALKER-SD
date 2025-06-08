# -*- coding: utf-8 -*-
# @Date     : 2025/05/10
# @Author   : Getsum
# @File     : get_rank.py
# @Github   : https://github.com/getsum-zero

from argparse import ArgumentParser
import os
import json
import logging
import yaml
import numpy as np
from tqdm import tqdm
from rank_base import UtteranceCluster, Utterance, random_call, Segment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('log.txt')),
    ]
)

parser = ArgumentParser(add_help=True)
parser.add_argument('--utterance_id', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--samples_nums', type=int)   
parser.add_argument('--cutting_type', choices=['whisper', 'pyannote_vad', 'noncut', 'direct_truncation'], default='noncut',)   
args = parser.parse_args()


def normalize(prob):
    prob = np.array(prob) / np.sum(prob)
    return prob



class UtteranceRanks():
    def __init__(self, utt_list, conf, pipeline):
        self.utt_list = utt_list
        self.conf = conf
        self.interval = normalize(conf['interval_ratio'])  # silence, overlap(2-4)

        self.silence_call = random_call(conf['silence'])
        self.overlap_call = {key-2:random_call(value) for key, value in conf['overlap'].items()}
        self.pipeline = pipeline['pipline']
        self.pipeline_type = pipeline['type']
        self.cutting_max_length = pipeline['cutting_max_length']

    def _cut(self, iter) -> Segment:

        if self.pipeline_type == 'whisper':
            # default is the first utt
            utt = self.utterance[iter].utt_list[0]
            idx = utt.idx
            waves = self.utt_list[idx]['path']
            segments, _ = self.pipeline.transcribe(waves, beam_size=5, word_timestamps=True,) #language="en")
            texts = ''
            for segment in segments:
                for word in segment.words:
                    if word.end > self.cutting_max_length: 
                        return Segment(idx, 0, word.start, texts)
                    texts += word.word + ' '
            return Segment(idx, 0, utt.duration)  # return the whole utterance
                
        elif self.pipeline_type == 'pyannote_vad':
            split_segs = []
            for utt in self.utterance[iter].utt_list:
                idx = utt.idx
                waves = self.utt_list[idx]['path']
                vad_res = self.pipeline(waves)
                for seg in vad_res.get_timeline():
                    split_segs.append(Segment(idx, seg.start, seg.end - seg.start, "<Unkown>"))

                split_segs.sort(key=lambda x: x.cutting_length)
                assert len(split_segs) > 0, "No segments found after Vad."
                return split_segs[0]
            
        elif self.pipeline_type == 'direct_truncation':
            utt = self.utterance[iter].utt_list[0]
            idx = utt.idx
            waves = self.utt_list[idx]['path']
            if utt.duration > self.cutting_max_length:
                return Segment(idx, 0, self.cutting_max_length, "<Unkown>")
            else:
                return Segment(idx, 0, utt.duration)
        else: 
            return None




    def order(self, idex):
        '''
            Order the utterance list by the idex
        '''
        self.utterance = [ 
            Utterance(
                i,
                self.utt_list[i]['duration'], 
                self.utt_list[i]['speaker_id'], 
                None
            ) 
            for i in idex
        ]
    
    def combine(self):
        '''
            Combine the adjacent utterance of the same speaker
            Ensure that adjacent utts are from different speakers 
        '''
        temp_list = []
        now_utt = UtteranceCluster(self.utterance[0])
        for i in self.utterance[1:]:
            if now_utt.speaker_id == i.speaker_id:
                silence_len = self.silence_call()
                now_utt.combine(i, silence_len)
            else:
                temp_list.append(now_utt)
                now_utt = UtteranceCluster(i)
        temp_list.append(now_utt)
        self.utterance = temp_list

        # for i in self.utterance:
        #     print(f"speaker_id: {i.speaker_id}, duration: {i.duration}, head_time: {i.head_time}")
        #     for j in i.utt_list:
        #         print(f"    {j.speaker_id}, {j.duration}, {j.head_time}")
    
    def insert(self):
        first_utt = self.silence_call() if np.random.rand() < self.interval[0] else 0.0
        self.utterance[0].head_time = first_utt
        
        
        # end_overlap_time: The end of the previous overlap segment ensures that the current overlap does not overlap with the previous overlap; 
        #               At the same time, it can avoid overlaps with the same speaker.
        # length: The length of the current waves
        end_overlap_time, length = first_utt, first_utt + self.utterance[0].duration
        last_speaker = self.utterance[0].speaker_id
        iter, iter_end = 1, len(self.utterance)
        while iter < iter_end:
            silence_or_overlap = np.random.choice(len(self.interval), p=self.interval)
            if silence_or_overlap == 0 or last_speaker == self.utterance[iter].speaker_id:     # silence
                self.utterance[iter].head_time = length + self.silence_call()
                end_overlap_time = self.utterance[iter].head_time
                length = self.utterance[iter].head_time + self.utterance[iter].duration
                last_speaker = self.utterance[iter].speaker_id
                iter += 1
            else:
                silence_or_overlap = min(silence_or_overlap, iter_end - iter)  # Make sure there is enough utt for overlap
                # Avoid overlap between the same speakers
                spks = [last_speaker]
                for i in range(silence_or_overlap):
                    if self.utterance[iter+i].speaker_id in spks:
                        silence_or_overlap = i
                        break
                    spks.append(self.utterance[iter+i].speaker_id)
                    
                tmp_length = length
                max_cover = length - end_overlap_time

                if 'libri_5639' in spks:
                    p = 1
                else: p = 0
                if p:
                    print("=======")
                    print(f"iter: {iter}, silence_or_overlap: {silence_or_overlap}, length: {length}, end_overlap_time: {end_overlap_time}, last_speaker: {last_speaker}")

                max_id = -1
                for i in range(silence_or_overlap):
                    # Make sure overlap>2 does not cause additional overlap
                    # |____________________|                   utt1
                    #                 |______________|         utt2
                    #                    |______________|      utt3
                    # utt2 and utt3 will cause additional overlap, which is not what we want.
                    # Here, we use vad to intercept utt2 and utt3, and select the shortest segments for overlap
                    if i > 0:
                        seg = self._cut(iter+i)
                        if seg:
                            self.utterance[iter+i].set_cutting_flag(seg)


                    overlap_len = min(self.overlap_call[i](), max_cover)
                    self.utterance[iter+i].head_time = tmp_length - overlap_len

                    if self.utterance[iter+i].head_time + self.utterance[iter+i].duration > length:
                        end_overlap_time = length
                        last_speaker = self.utterance[iter+i].speaker_id
                        length = self.utterance[iter+i].head_time + self.utterance[iter+i].duration
                        max_id = i

                    if p:
                        print(length, end_overlap_time, last_speaker)
                        print(f"iter: {iter+i}, head_time: {self.utterance[iter+i].head_time}, duration: {self.utterance[iter+i].duration}, "
                              f"speaker_id: {self.utterance[iter+i].speaker_id}, overlap_len: {overlap_len}, end_overlap_time: {end_overlap_time}")

                
                   # The overlap segment does not exceed the original length
                end_overlap_time = max(
                                        end_overlap_time, 
                                        np.max([self.utterance[iter+i].head_time + self.utterance[iter+i].duration 
                                                if i != max_id else end_overlap_time
                                                for i in range(silence_or_overlap)
                                        ])
                                )

                iter += silence_or_overlap
        
        # for i in self.utterance:
        #     print(f"speaker_id: {i.speaker_id}, duration: {i.duration}, head_time: {i.head_time}")

    def get_headtimes(self):
        '''
            Get the head time of each utterance
        '''
        start_times = []
        cutting_timestamp = []
        cutting_texts = []
        for i in self.utterance:

            if i.cutting_flag == False:
                start_times.append(i.head_time)
                cutting_timestamp.append((0, i.utt_list[0].duration))
                cutting_texts.append('<whole>')
                nowtime = i.head_time + i.utt_list[0].duration
                for j in i.utt_list[1:]:
                    start_times.append(j.head_time + nowtime)
                    cutting_timestamp.append((0, j.duration))
                    cutting_texts.append('<whole>')
                    nowtime = nowtime + j.head_time + j.duration
                assert np.abs(nowtime - i.head_time - i.duration) < 1e-3, f"nowtime: {nowtime}, head_time: {i.head_time}, duration: {i.duration}"


            else:
                for j in i.utt_list:
                    if j.head_time is None:
                        start_times.append(None)
                        cutting_timestamp.append((-1,-1))
                        cutting_texts.append("<drop>")
                        # Todo
                        # 这里会直接将多的utt抛弃,如果这个cluster较大,抛弃的utt会过大
                        # 交换机制能一定程度抑制
                    else:
                        start_times.append(i.head_time)
                        cutting_timestamp.append((j.head_time, i.duration))
                        cutting_texts.append(i.text)

        return [start_times, cutting_timestamp, cutting_texts]




def discussion_meeting(utt_list, conf, pipeline):
    """
    Generate a discussion meeting
        Completely random speech
    """

    # 1. init the UtteranceRanks 
    ranks = UtteranceRanks(utt_list, conf, pipeline)

    # 2. shuffle the utterance list
    idex = np.random.permutation(np.arange(len(utt_list)).astype(np.int32))
    ranks.order(idex)

    # 3. merge adjacent utterance of the same speaker
    ranks.combine()

    # 4. add silence or overlap between adjacent utts 
    ranks.insert()
    
    # 4. get start time for each utterance
    start_times = ranks.get_headtimes()

    return idex, start_times

def presentation_meeting(utt_list, conf, split_num, pipeline):
    """
    Generate a presentation meeting
    """

    ranks = UtteranceRanks(utt_list, conf, pipeline)

    idex = np.ones(len(utt_list), dtype=np.int32) * -1
    main_speaker_idx = np.arange(split_num).tolist()
    other_speaker_idx = np.arange(split_num, len(utt_list)).tolist()

    # Select some main utterances as secondary utterances to participate in the speech
    main_shuffle = np.random.choice(main_speaker_idx, int(split_num * conf['main_speaker_shuffle']), replace=False).tolist()
    other_speaker_idx = other_speaker_idx + main_shuffle
    main_speaker_idx = [i for i in main_speaker_idx if i not in main_shuffle]

    # get order
    main_speaker_idx.sort(key=lambda x: utt_list[x]['speaker_id'])
    np.random.shuffle(other_speaker_idx)
    others_insert_main_id = np.random.choice(len(utt_list), len(other_speaker_idx), replace=False)
    idex[others_insert_main_id] = other_speaker_idx
    idex[idex == -1] = main_speaker_idx
    
    ranks.order(idex)
    ranks.combine()
    ranks.insert()
    start_times = ranks.get_headtimes()
    return idex, start_times

def interview_meeting(utt_list, conf, split_num, pipeline):
    ranks = UtteranceRanks(utt_list, conf, pipeline)

    idex = np.ones(len(utt_list), dtype=np.int32) * -1
    interview_idx = np.arange(split_num).tolist()
    other_speaker_idx = np.arange(split_num, len(utt_list)).tolist()
    other_speaker_idx.sort(key=lambda x: utt_list[x]['speaker_id'])

    interview_insert_others_id = np.random.choice(len(utt_list), len(interview_idx), replace=False)
    idex[interview_insert_others_id] = interview_idx
    idex[idex == -1] = other_speaker_idx
    
    ranks.order(idex)
    ranks.combine()
    ranks.insert()
    start_times = ranks.get_headtimes()
    return idex, start_times


if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)
    config = yaml.safe_load(open(args.config, 'r'))

    # load vad for cut
    if args.cutting_type == 'pyannote_vad':
        from pyannote.audio.pipelines import VoiceActivityDetection
        from pyannote.audio import Model
        model = Model.from_pretrained(config['vad_pretrained_model'])
        pipeline = VoiceActivityDetection(segmentation=model)
        HYPER_PARAMETERS = {
            "onset": 0.5, "offset": 0.5,
            "min_duration_on": 0.5,  
            "min_duration_off": 0.0 
        }
        pipeline.instantiate(HYPER_PARAMETERS)
    elif args.cutting_type == 'whisper':
        from faster_whisper import WhisperModel
        pipeline = WhisperModel(config['whiper_model'], device="cuda", compute_type="float16")
    else:
        pipeline = None
    pipeline = {'pipline': pipeline, 'type': args.cutting_type, 'cutting_max_length': config['cutting_max_length']}

    # seed
    np.random.seed(config['seed'])

    pbar = tqdm(total=args.samples_nums, desc='Processing', unit='samples')
    iter = 0
    with open(args.utterance_id, 'r') as f:
        for line in f:
            line = line.strip()
            sample = json.loads(line)
            
            if sample['meeting_type'] == 'presentation':
                idex, res = presentation_meeting(sample['utt_list'], config['meeting']['presentation'], sample['split_num'], pipeline)
            elif sample['meeting_type'] == 'interview':
                idex, res = interview_meeting(sample['utt_list'], config['meeting']['interview'], sample['split_num'], pipeline)
            elif sample['meeting_type'] == 'discussion':
                idex, res = discussion_meeting(sample['utt_list'], config['meeting']['discussion'], pipeline)
            
            start_times, cutting_timestamp, cutting_texts = res
            assert len(idex) == len(start_times) == len(cutting_timestamp) == len(cutting_texts), "The length of idex, start_times and cutting_timestamp must be the same."

            # rttm type recording
            #   start_time    end_time    speaker_id    utterance_id 
            speaker_logging = os.path.join(args.output_dir, f'{iter:07d}.list')   
            with open(speaker_logging, 'w', encoding='utf-8') as f_out: 
                for id, start_time, timestamps, cutting_text in zip(idex, start_times, cutting_timestamp, cutting_texts):

                    cut_start, duration = timestamps
                    if cut_start == -1:  # For multiple (greater than 2) overlaps, the utt is dropped off
                        continue
                    Transcription = sample['utt_list'][id]['Transcription'] if cutting_text == '<whole>' else cutting_text

                    speaker_id = sample['utt_list'][id]['speaker_id']
                    utterance_id = sample['utt_list'][id]['path']
                    if cutting_text == '<whole>':  assert duration == sample['utt_list'][id]['duration']
                    f_out.write(f"{start_time:.3f} {start_time+duration:.3f} {speaker_id} {utterance_id} {cut_start:.3f} [{Transcription}]\n")

            pbar.update(1)
            iter += 1
    