

# 目标下载目录

base_root=$1
mkdir -p $base_root
cd $base_root


# 设置下载目标目录
TARGET_DIR="LibriSpeech"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# LibriSpeech 数据子集
DATASETS=(
  "train-clean-100"
  "train-clean-360"
  "train-other-500"
  "dev-clean"
  "dev-other"
  "test-clean"
  "test-other"
)

# 下载地址前缀
BASE_URL="https://www.openslr.org/resources/12"

# 最大并发数
MAX_JOBS=4

# 当前后台任务数
job_count=0

download_and_extract() {
  dataset=$1
  url="${BASE_URL}/${dataset}.tar.gz"

  echo "⬇️ 开始下载: $dataset"
  wget -c "$url" -O "${dataset}.tar.gz"

  if [ -f "${dataset}.tar.gz" ]; then
    echo "📦 解压: ${dataset}.tar.gz"
    tar -xzf "${dataset}.tar.gz"
    echo "✅ 完成: $dataset"
  else
    echo "❌ 错误: ${dataset}.tar.gz 未找到"
  fi
}

for dataset in "${DATASETS[@]}"; do
  # 启动并发任务
  download_and_extract "$dataset" &

  ((job_count++))
  # 控制最大并发数
  if (( job_count >= MAX_JOBS )); then
    wait -n  # 等待任一任务完成后再继续
    ((job_count--))
  fi
done

# 等待所有后台任务完成
wait

echo "🎉 所有数据集处理完成。"

