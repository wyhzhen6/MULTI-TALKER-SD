
base_root=$1
mkdir -p $base_root
cd $base_root

# 创建存储目录
TARGET_DIR="AISHELL-1"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# 下载地址（来自 openslr.org）
BASE_URL="https://www.openslr.org/resources/33"
FILES=(
  "data_aishell.tgz"
  "resource_aishell.tgz"
)

# 最大并发数
MAX_JOBS=2
job_count=0

download_and_extract() {
  file=$1
  echo "⬇️ 正在下载: $file"
  wget -c "${BASE_URL}/${file}" -O "$file"

  if [ -f "$file" ]; then
    echo "📦 正在解压: $file"
    tar -xzf "$file"
    echo "✅ 完成: $file"
  else
    echo "❌ 错误: 文件 $file 未找到"
  fi
}

for file in "${FILES[@]}"; do
  download_and_extract "$file" &
  ((job_count++))
  if (( job_count >= MAX_JOBS )); then
    wait -n
    ((job_count--))
  fi
done

wait
echo "🎉 AISHELL-1 数据集下载并解压完成。目录：$(pwd)"


find "$TARGET_DIR"/data_aishell/wav -maxdepth 1 -type f -name "*.tar.gz" | while read -r file; do
  echo "解压: $file"
  tar -xzvf "$file" -C "$TARGET_DIR"/data_aishell/wav
done


