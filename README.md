# Multi-Talker-SD: Large-Scale Bilingual Meeting Diarization Dataset

This project provides Multi-Talker-SD, a large-scale bilingual (English–Mandarin) multi-speaker meeting dataset designed to support research on speaker diarization and meeting transcription. The dataset comprises 1,000 simulated meetings, each with 10–30 participants and average conversation durations of around 20 minutes, capturing realistic speaker overlap, turn-taking patterns, and multilingual interactions. The audio is synthesized using utterances from AIShell-1 and LibriSpeech, combined with reverberation and noise injection to generate high-fidelity, multi-speaker recordings. The dataset also includes detailed speaker metadata, such as gender, language, session type, and utterance timing, enabling controlled experiments and ablation studies.

The repository is organized into two main components:

**Frontend**: Speaker Logging, including metadata management and diarization-ready segmentation.

**Backend**: Acoustic Model Construction, including synthesis pipeline, noise/reverberation modeling, and audio generation.

---

## Usage

### Download Data


Use the download scripts in the script directory to obtain the corresponding datasets. For example:

```
bash script/download_librispeech.sh <your_save_dir>
```
Point-source and diffuse-field noise data can be downloaded from the following link: https://1drv.ms/u/c/969dad2e7ff5ab41/EcV68xcR9pVHsd3yNWSTzxkBkKvfLwTQsOluZJOnzf1GFA?e=OnfDv5

* Core Environment Dependencies：
    * faster_whisper==1.1.1 （core component; it is recommended to install all related libraries according to faster_whisper）
    * soundfile
    * tqdm
    * torch
    * torchaudio
      
You can quickly set up a Conda environment using the provided requirements.txt：
```
conda create -n diarization_env python=3.10 -y
conda activate diarization_env
pip install -r requirements.txt
```



* Run `run.sh`
Key parameters:
```
    exp_dir                # Directory to save the generated WAV files
    librispeech_dir        # Path to LibriSpeech; ensure SPEAKERS.TXT exists in this directory
    aishell_1_dir          # Path to AIShell-1; ensure resource_aishell folder exists with speaker.info and lexicon.txt
    point_noise_path       # Path to point-source noise data
    diffuse_noise_path     # Path to diffuse-field noise data
```

* Key Configuration in config/config.yaml

  iteration: Number of iterations. The program will attempt this many iterations, generating at most one WAV file per iteration. The total number of generated WAV files will be less than or equal to iteration.

  max_examples: Maximum number of WAV files to generate. To generate an exact number of WAV files, adjust this parameter. For example, setting it to 100 will generate exactly 100 WAV files. It must be smaller than iteration; otherwise, it will be ignored.

## Resources

* 📂 **Dataset Availability Notice:** This repository is released in anonymized form for the purpose of peer review.
Due to the double-blind review policy, the full Multi-Talker-SD dataset is not publicly available at this stage.

The complete dataset, along with detailed documentation and access instructions, will be released upon acceptance of the paper.

