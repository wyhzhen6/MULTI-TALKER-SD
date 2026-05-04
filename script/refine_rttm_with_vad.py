import os
import glob
import argparse
import yaml
import torch
from pyannote.core import Annotation, Segment
from pathlib import Path

def load_rttm(rttm_path):
    """加载原始 RTTM，返回 pyannote.core.Annotation 对象"""
    annotation = Annotation()
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: 
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation

def save_rttm(annotation, uri, output_path):
    """保存修正后的 Annotation 为 RTTM 格式"""
    with open(output_path, 'w') as f:
        # itersegments 保证按时间顺序输出
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            f.write(f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

def init_vad_pipeline(config_path=None, hf_token=None):
    """初始化 VAD Pipeline"""
    from pyannote.audio.pipelines import VoiceActivityDetection
    from pyannote.audio import Model

    print("Loading pyannote/segmentation-3.0 from HuggingFace...")
    # 按照 VAD.py 的方式直接加载 HuggingFace 上的 pyannote/segmentation-3.0
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    
    HYPER_PARAMETERS = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    return pipeline

def main():
    parser = argparse.ArgumentParser(description="Refine RTTM using VAD to remove silence")
    parser.add_argument('--wav_dir', required=True, help='Directory containing the .wav files')
    parser.add_argument('--rttm_dir', required=True, help='Directory containing the original .rttm files')
    parser.add_argument('--output_dir', required=True, help='Directory to save the refined .rttm files')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml (optional, used to load local VAD model)')
    parser.add_argument('--hf_token', type=str, default="YOUR_HF_TOKEN_HERE", help='HuggingFace Auth Token')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = init_vad_pipeline(args.config, args.hf_token)

    rttm_files = glob.glob(os.path.join(args.rttm_dir, "*.rttm"))
    print(f"Found {len(rttm_files)} RTTM files to process.")

    for rttm_path in rttm_files:
        uri = Path(rttm_path).stem
        
        # 尝试匹配音频文件
        wav_path = os.path.join(args.wav_dir, f"{uri}.wav")
        if not os.path.exists(wav_path):
            print(f"[Warning] Audio not found: {wav_path}, skipping {uri}...")
            continue
        
        print(f"Processing {uri}...")
        
        # 1. 跑 VAD，获取真实的语音活动 Timeline
        vad_output = pipeline(wav_path)
        # support() 可以把相邻或重叠的语音段合并为一个干净的 Timeline
        vad_timeline = vad_output.get_timeline().support() 
        
        # 2. 读取原始的 RTTM
        orig_annotation = load_rttm(rttm_path)
        
        # 3. 取交集：只保留存在于 VAD Timeline 中的标签段
        refined_annotation = Annotation(uri=uri)
        for segment, track, speaker in orig_annotation.itertracks(yield_label=True):
            # 将原始的 segment 与 VAD 的结果取交集
            # vad_timeline.crop(segment) 会返回在 segment 范围内的所有 VAD 段落
            cropped_timeline = vad_timeline.crop(segment, mode='intersection')
            for cropped_seg in cropped_timeline:
                refined_annotation[cropped_seg] = speaker
                
        # 4. 保存修正后的 RTTM
        output_path = os.path.join(args.output_dir, f"{uri}.rttm")
        save_rttm(refined_annotation, uri, output_path)

    print("All done!")

if __name__ == '__main__':
    main()
