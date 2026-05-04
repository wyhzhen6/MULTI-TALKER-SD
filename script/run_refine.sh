#!/bin/bash
#SBATCH --job-name=refine_rttm
#SBATCH --output=refine_rttm_%j.log
#SBATCH --error=refine_rttm_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# 如果你的环境中需要 conda，请将下面的路径改为你自己的环境，并取消注释
source ~/.bashrc
conda activate VBx

# 基础目录配置
PROJECT_DIR="/home3/yihao/Research/Code/Large-scale-diarization-dataset"
TEST_DIR="${PROJECT_DIR}/exp/dataset-test-basebend_final/test"

# 输入/输出目录
WAV_DIR="${TEST_DIR}/wavs/clean"
RTTM_DIR="${TEST_DIR}/GT"
OUTPUT_DIR="${TEST_DIR}/GT_refined"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
SCRIPT_PATH="${PROJECT_DIR}/script/refine_rttm_with_vad.py"

echo "==================================================="
echo "Start refining RTTM using VAD..."
echo "Date: $(date)"
echo "WAV_DIR: ${WAV_DIR}"
echo "RTTM_DIR: ${RTTM_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "==================================================="

# 确保输出目录存在
mkdir -p ${OUTPUT_DIR}

# 运行 Python 修正脚本
python ${SCRIPT_PATH} \
    --wav_dir ${WAV_DIR} \
    --rttm_dir ${RTTM_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --config ${CONFIG_FILE}

echo "==================================================="
echo "Done!"
echo "Date: $(date)"
echo "==================================================="
