import os
import glob
import argparse
from pathlib import Path

def convert_list_to_rttm(list_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    list_files = glob.glob(os.path.join(list_dir, '*.list'))
    
    print(f"Found {len(list_files)} .list files in {list_dir} to convert to RTTM.")
    
    for list_path in list_files:
        recording_id = Path(list_path).stem
        rttm_path = os.path.join(output_dir, f"{recording_id}.rttm")
        
        with open(list_path, 'r') as f_in, open(rttm_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(maxsplit=5)
                if len(parts) < 4:
                    continue
                start_time = float(parts[0])
                end_time = float(parts[1])
                speaker_id = parts[2]
                duration = end_time - start_time
                
                if duration > 0:
                    rttm_line = f"SPEAKER {recording_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                    f_out.write(rttm_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .list files to .rttm")
    parser.add_argument('--list_dir', type=str, required=True, help="Directory containing .list files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save .rttm files")
    args = parser.parse_args()
    
    convert_list_to_rttm(args.list_dir, args.output_dir)
