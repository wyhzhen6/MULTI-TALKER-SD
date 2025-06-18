#!/bin/bash

set -eou pipefail

log() {
  	# This function is from espnet
  	local fname=${BASH_SOURCE[1]##*/}
  	echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=2


# ================= Fill in according to actual =================
exp_dir=!				# where .wav to store
librispeech_dir=!		# 
aishell_1_dir=!			# Usually named data_aishell




# ============================ get metadata ============================ #

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
			--transcript		$aishell_1_dir/data_aishell/transcript/aishell_transcript_v0.8.txt 
	log "Finished generating metadata for Aishell-1 dataset"

fi

# ============================ Divide the train, eval and test set ============================ #

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
	
	for lang in Chinese English; do 
		ls metadata/$lang | while read dataset; do
			echo metadata/$lang/$dataset
			python src/metadata/divide_sub_set.py --from_dir metadata/$lang/$dataset \
				--base_dir 	metadata \
				--subset train test dev \
				--dataset $lang/$dataset \
				--radio 0.8 0.1 0.1
		done
	done
fi



subsets=(train test dev)
# ============================ speaker logging ============================ #
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
	for subset in "${subsets[@]}"; do
		subset_dir=metadata/$subset
		log "Getting speaker ids in $subset_dir"
		python src/speaker_log/get_speaker_id.py \
				--metadata_dir $subset_dir \
				--output_dir $subset_dir \
				--config config/config.yaml
		log "Finished getting speaker ids"

		log "Getting utterance ids"
		python src/speaker_log/get_utterance_id.py \
				--speaker_id $subset_dir/speaker_id.list \
				--output_dir $subset_dir \
				--config config/config.yaml
		log "Finished getting utterance ids"

		log "Speaker logging: set ranking, silence, and overlap"
		samples_nums=$((`cat $subset_dir/utterance_id.list | wc -l`))
		# python src/speaker_log/get_rank.py \
		# 		--utterance_id $subset_dir/utterance_id.list \
		# 		--output_dir $subset_dir/samples \
		# 		--samples_nums $samples_nums \
		# 		--config config/config.yaml \
		# 		--cutting_type whisper  	# direct_truncation
	done
fi


subsets=(train test dev)
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
	for subset in "${subsets[@]}"; do
		subset_dir=metadata/$subset
		mkdir -p $subset_dir/wavs
		ls $subset_dir/samples | while read line; do
			log "Processing ${line}"
			python src/speaker_log/create_wav.py \
				--logging_file $subset_dir/samples/${line} \
				--output_dir $subset_dir/wavs 
		done
		mv $subset_dir/wavs $exp_dir
		mv $subset_dir/samples $exp_dir
		cp config.yaml $exp_dir
fi