#!/bin/bash
# 下载上游数据集原始文件到 data/raw/ 并校验指纹（公开版不分发数据本体）。
# 来源：ModelScope 数据集 zhangmingme/medical-record-diagnosis-chinese
#       （上传者声明 Apache-2.0；出处说明与下载实录见 data/PROVENANCE.md）
# 用法：bash data/fetch_raw.sh        （在仓根执行；已存在的文件跳过，已存在的错误文件请先删除）
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://modelscope.cn/datasets/zhangmingme/medical-record-diagnosis-chinese/resolve/master"
mkdir -p data/raw
for f in medical-record-diagnosis.json medical-record-diagnosis-sft.jsonl; do
  if [ -s "data/raw/$f" ]; then echo "[skip] data/raw/$f 已存在"; continue; fi
  echo "[fetch] $f"
  curl -fL --retry 3 --progress-bar -o "data/raw/$f.part" "$BASE/$f"
  mv "data/raw/$f.part" "data/raw/$f"
done
sha256sum -c data/raw/SHA256SUMS
