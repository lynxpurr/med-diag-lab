#!/bin/bash
# MED-D D3 评测跑批：三测点 × 五集（val-reg/test-ood/test-rob/neg/negn）串行，幂等（score 在即跳）
# 冠军基线读数复用 MED-B/C 报告（val-reg 0.800 / fmt 80/80 / cov 0.976 / ood 0.133 /
#   rob 20/20 / minimal 20/20 / neg 19/20 / negn 10/10）——不重推理。
# 用法：bash tools/run_med_d3.sh          # 三测点全跑（缺哪个补哪个）
#   环境注：14B 4bit 推理 batch1 ≈ ~10GB，第三方上卡 250-600MiB 可容忍（NEED 11000）。
set -euo pipefail
cd "$(dirname "$0")/.."

SETS="val-reg test-ood test-rob neg negn"
declare -A CKPT=( [dpo-s1]=output/med-d/14b-dpo-s1/checkpoint-1
                  [dpo-s3]=output/med-d/14b-dpo-s3/checkpoint-3
                  [dpo-s9]=output/med-d/14b-dpo-s9/checkpoint-9 )
NEED=11000
OUT=output/med-d

for tag in dpo-s1 dpo-s3 dpo-s9; do
  ckpt=${CKPT[$tag]}
  test -f "$ckpt/args.json" || { echo "[skip] $ckpt 不在（臂未完成）"; continue; }
  for s in $SETS; do
    if [ -e "$OUT/score-$tag-$s.json" ]; then echo "[skip] $tag/$s 已有 score"; continue; fi
    if [ ! -e "$OUT/$tag-$s.jsonl" ]; then
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      if [ "$FREE" -lt "$NEED" ]; then echo "[wait] $tag/$s 需 ${NEED}MiB 现 ${FREE}MiB，等 60s"; sleep 60; fi
      echo "=== infer $tag/$s $(date +%T) ==="
      python3 tools/run_eval_med.py infer "$tag" "$s" --adapters "$ckpt" --out "$OUT"
    fi
    echo "=== score $tag/$s ==="
    python3 tools/run_eval_med.py score "$tag" "$s" --out "$OUT"
  done
done
echo "=== 全部完成，聚合主表 ==="
python3 tools/run_eval_med.py table --out "$OUT"
