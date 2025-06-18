

base_root=$1
mkdir -p $base_root
cd $base_root


TARGET_DIR="VCTK"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

URL="https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
FILENAME="VCTK-Corpus-0.92.zip"

echo "⬇️ 正在下载 VCTK 数据集 (0.92)…"
wget -c "$URL" -O "$FILENAME"

if [ -f "$FILENAME" ]; then
  echo "📦 下载完成，开始解压..."
  unzip -q "$FILENAME"
  echo "✅ 解压完成"
else
  echo "❌ 下载失败：未找到 $FILENAME"
  exit 1
fi

echo "🎉 所有任务完成，数据目录：$(pwd)"
