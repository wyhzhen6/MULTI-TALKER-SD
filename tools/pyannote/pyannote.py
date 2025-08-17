

from pyannote.audio import Pipeline
import argparse
import torch
import torchaudio
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_wavs", type=str)
parser.add_argument("--output_rttm", type=str)
args = parser.parse_args()

start_times = time.time()

pipeline = Pipeline.from_pretrained( "pyannote/speaker-diarization-3.1",
        use_auth_token="xxxxx")

print(f"load model: {time.time()-start_times}")

pipeline.to(torch.device("cuda"))

waveform, sample_rate = torchaudio.load(args.input_wavs)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# dump the diarization output to disk using RTTM format
with open(args.output_rttm, "w") as rttm:
    diarization.write_rttm(rttm)
input_wavs = args.input_wavs
print(f"deal with {input_wavs} : {time.time()-start_times}")
