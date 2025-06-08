# -*- coding: utf-8 -*-
# @Date     : 2025/05/19
# @Author   : Getsum
# @File     : rank_base.py
# @Github   : https://github.com/getsum-zero

import numpy as np
from dataclasses import dataclass

@dataclass
class Segment:
    utt_idx: int
    cutting_start_time: float
    cutting_length: float
    text: str = None



class Utterance:
    def __init__(self, idx, duration, speaker_id, head_time, text=None):
        self.idx = idx
        self.duration = duration
        self.speaker_id = speaker_id
        self.head_time = head_time
    

class UtteranceCluster():
    def __init__(self, utt):
        self.duration = utt.duration
        self.speaker_id = utt.speaker_id
        self.head_time = utt.head_time
        self.cutting_flag = False
        self.text = None
        
        self.utt_list = [utt]

    def set_cutting_flag(self, seg: Segment):
        '''
            Set the cutting flag for the utterance cluster
            seg: (idx, cutting_start_time, cutting_length)
        '''
        self.cutting_flag = True
        self.duration = seg.cutting_length
        self.text = seg.text
        for id in range(len(self.utt_list)):
            self.utt_list[id].head_time = None if self.utt_list[id].idx != seg.utt_idx else seg.cutting_start_time

    def combine(self, utt, silence_len):
        '''
            Combine the utterance with the silence length
        '''
        utt.head_time = silence_len
        self.utt_list.append(utt)
        self.duration += utt.duration + silence_len


class random_call():
    def __init__(self, ranges):
        ranges, distru = ranges.split(':')
        if distru == 'unif':
            mins, maxs = ranges.split('-')
            self.mean, self.std = float(mins), float(maxs) - float(mins)
            self.rand = self.unif
        elif distru == 'norm':
            l, r = ranges.split('-')
            self.mean, self.std = float(l), float(r)
            self.rand = self.norm
        else:
            return NotImplementedError
            


    def unif(self):
        return np.random.rand() * self.std + self.mean
    def norm(self):
        return np.random.randn() * self.std + self.mean

    def __call__(self, digits=3, not_negative=True):
        '''
            if not_negative is True,
            it will be recursive call until it is greater than 0
        '''
        res = round(self.rand(), digits)
        return res if not_negative and res > 0 else self.__call__(digits, not_negative)