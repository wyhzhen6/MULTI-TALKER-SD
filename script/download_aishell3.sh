

base_root=$1
mkdir -p $base_root
cd $base_root

# 设置下载目录
TARGET_DIR="AISHELL-3"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# 下载地址前缀




BASE_URL="https://www.openslr.org/resources/93"
FILES=(
  "data_aishell3.tgz"
)

# 最大并发数
MAX_JOBS=1
job_count=0

# 下载并解压函数
download_and_extract() {
  file=$1
  echo "⬇️ 正在下载: $file"
  wget -c "${BASE_URL}/${file}" -O "$file"

  if [ -f "$file" ]; then
    echo "📦 正在解压: $file"
    tar -xzf "$file"
    echo "✅ 解压完成: $file"
  else
    echo "❌ 错误: 文件 $file 未找到"
  fi
}

# 主循环：并发下载
for file in "${FILES[@]}"; do
  download_and_extract "$file" &
  ((job_count++))

  if (( job_count >= MAX_JOBS )); then
    wait -n
    ((job_count--))
  fi
done

wait
echo "🎉 AISHELL-3 数据集下载与解压已完成，路径：$(pwd)"
