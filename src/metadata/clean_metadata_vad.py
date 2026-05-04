import os
import glob
import json
import argparse
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from tqdm import tqdm

def process_metadata(metadata_dir, hf_token):
    print(f"Loading pyannote/segmentation-3.0 for VAD cleaning on {metadata_dir}...")
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    # Only process utterance-level json files (they contain individual utt records with 'path' key)
    # These are located in utterance/ subdirectories; speaker-level jsons only have pointers to them
    json_files = glob.glob(os.path.join(metadata_dir, '**/utterance/*.json'), recursive=True)
    print(f"Found {len(json_files)} utterance JSON files to process.")

    total_utts = 0
    updated_utts = 0

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r') as f:
            try:
                utts = json.load(f)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
                
        needs_update = False
        
        for utt in utts:
            total_utts += 1
            if 'vad_start' not in utt:
                audio_path = utt['path']
                if not os.path.exists(audio_path):
                    # fallback to 0 if file not found
                    utt['vad_start'] = 0.0
                    needs_update = True
                    continue
                
                try:
                    vad_res = pipeline(audio_path)
                    timeline = vad_res.get_timeline().support()
                    
                    if len(timeline) > 0:
                        first_start = timeline[0].start
                        last_end = timeline[-1].end
                        
                        utt['vad_start'] = first_start
                        utt['duration'] = last_end - first_start
                    else:
                        utt['vad_start'] = 0.0
                        # duration keeps its original value
                        
                    needs_update = True
                    updated_utts += 1
                except Exception as e:
                    print(f"Failed to process audio {audio_path}: {e}")
                    utt['vad_start'] = 0.0
                    needs_update = True
                    
        if needs_update:
            with open(json_file, 'w') as f:
                json.dump(utts, f, ensure_ascii=False, indent=4)

    print(f"Finished. Total utterances checked: {total_utts}, Updated: {updated_utts}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean original audio duration using VAD")
    parser.add_argument('--metadata_dir', type=str, default='metadata', help='Base directory containing JSON metadata')
    parser.add_argument('--hf_token', type=str, default="YOUR_HF_TOKEN_HERE", help='HuggingFace token')
    args = parser.parse_args()
    
    process_metadata(args.metadata_dir, args.hf_token)
