#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

basedir=
wav_dir=$basedir/wavs/noisy
res_dir=$basedir/results/pyannote/noisy/rttm/
mkdir -p $res_dir

for audio in $(ls $wav_dir)
do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    python pyannote.py --input_wavs $wav_dir/$filename.wav \
        --output_rttm $res_dir/$filename.rttm



done