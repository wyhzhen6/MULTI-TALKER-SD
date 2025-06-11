#!/bin/bash

set -eou pipefail

log() {
  	# This function is from espnet
  	local fname=${BASH_SOURCE[1]##*/}
  	echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=2
stop_stage=2


# ============================ get metadata ============================ #
# librispeech
# 1. Download the LibriSpeech dataset
    # wget -P data/LibriSpeech/ http://www.openslr.org/resources/12/dev-clean.tar.gz
    # tar -xzf data/LibriSpeech/dev-clean.tar.gz -C data/LibriSpeech/
# 2. Extract the metadata files

librispeech_dir=/mnt/e/wsl/data/LibriSpeech
aishell_1_dir=/mnt/e/wsl/data/aishell-1


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  	mkdir -p metadata/

	# =============================== English ============================== #
	lang_out=metadata/English
	log "Generating metadata for LibriSpeech dataset"
	python src/metadata/librispeech.py \
			--librispeech_dir 	$librispeech_dir \
			--output_dir 		$lang_out 
	log "Finished generating metadata for LibriSpeech dataset"


	# =============================== Chinese ============================== #
	lang_out=metadata/Chinese
	log "Generating metadata for Aishell-1 dataset"
	python src/metadata/aishell-1.py \
			--aishell_1_dir 	$aishell_1_dir \
			--output_dir 		$lang_out	\
			--transcript		$aishell_1_dir/transcript/aishell_transcript_v0.8.txt 
	log "Finished generating metadata for Aishell-1 dataset"



fi



# ============================ speaker logging ============================ #
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
	log "Getting speaker ids"
	python src/speaker_log/get_speaker_id.py \
			--metadata_dir metadata \
			--output_dir metadata \
			--config config/config.yaml
	log "Finished getting speaker ids"

	log "Getting utterance ids"
	python src/speaker_log/get_utterance_id.py \
			--speaker_id metadata/speaker_id.list \
			--output_dir metadata \
			--config config/config.yaml
	log "Finished getting utterance ids"

	log "Speaker logging: set ranking, silence, and overlap"
	samples_nums=$((`cat metadata/utterance_id.list | wc -l`))
	python src/speaker_log/get_rank.py \
			--utterance_id metadata/utterance_id.list \
			--output_dir metadata/samples \
			--samples_nums $samples_nums \
			--config config/config.yaml \
			--cutting_type whisper  # direct_truncation

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

	ls metadata/samples | while read line; do
		log "Processing ${line}"
		python src/speaker_log/create_wav.py \
			--logging_file metadata/samples/${line} \
			--output_dir metadata/wavs 
	done
fi