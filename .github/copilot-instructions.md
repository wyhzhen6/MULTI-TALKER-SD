# AI Coding Agent Instructions for Large-scale Diarization Dataset

## Project Architecture

**Two-phase pipeline**: Frontend speaker logging → Backend acoustic simulation
- `src/speaker_log/`: Selects speakers/utterances, manages timing, silence, and overlaps
- `src/acoustic/`: Simulates room acoustics using pyroomacoustics, adds point/diffuse noise
- `src/metadata/`: Processes LibriSpeech/AISHELL-1 datasets into structured JSON metadata

**Data flow**: Raw datasets → Metadata generation → Train/test/dev split → Speaker logging → Acoustic simulation → Labeled WAV files

## Key Workflows

**Main pipeline execution**:
```bash
# Configure paths in run.sh, then execute stages 0-3
./run.sh  # Stages: metadata → split → speaker logging → acoustic simulation
```

**HPC execution**:
```bash
sbatch slurmrun.sh  # Requests 4 GPUs, runs ./run.sh
```

**Configuration**: All parameters controlled via `config/config.yaml`:
- Speaker counts, genders, languages
- Utterance lengths, meeting types (discussion/presentation/interview)
- Silence/overlap intervals, noise mixing ratios

## Development Patterns

**CLI scripts**: Use argparse with `--metadata_dir`, `--config`, `--output_dir` pattern
```python
parser = ArgumentParser()
parser.add_argument('--metadata_dir', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--output_dir', type=str)
```

**Multiprocessing**: Parallel processing with `multiprocessing.Process` and `Manager`
```python
from multiprocessing import Process, Manager
counter = Value('i', 0)  # Shared counter across processes
```

**Audio processing**: soundfile for I/O, torch/torchaudio for ML operations
```python
import soundfile as sf
data, samplerate = sf.read(wav_path)
```

**Configuration parsing**: YAML with nested structures for complex parameters
```python
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
```

**Logging**: Structured logging to `log.txt` with timestamps
```python
logging.basicConfig(
    handlers=[logging.FileHandler('log.txt')],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Dependencies & Environment

**Core stack**: faster_whisper==1.1.1, torch, torchaudio, pyroomacoustics
**Environment**: Conda environment with `pip install -r requirements.txt`

**Dataset paths**: Configure absolute paths for LibriSpeech, AISHELL-1, noise datasets in `run.sh`

## Common Tasks

**Adding new dataset**: Create metadata processor in `src/metadata/`, update `run.sh` stage 0
**Modifying simulation**: Edit `config/config.yaml`, test with small `iteration` values
**Debugging audio**: Check generated WAVs in `exp/*/wavs/`, examine logging in `log.txt`

## File Organization

- `config/config.yaml`: All simulation parameters
- `src/speaker_log/get_rank.py`: Core speaker timing logic with whisper-based cutting
- `src/acoustic/simulate.py`: Room acoustics simulation with noise addition
- `script/download_*.sh`: Dataset download scripts</content>
<parameter name="filePath">/home3/yihao/Research/Code/Large-scale-diarization-dataset/.github/copilot-instructions.md