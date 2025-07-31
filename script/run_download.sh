#!/bin/bash

# 设置目标目录
BASE_DIR="./data"

# 调用下载脚本
#bash download_aishell1.sh "$BASE_DIR"
#bash download_librispeech.sh "$BASE_DIR"
bash download_vctk.sh "$BASE_DIR"