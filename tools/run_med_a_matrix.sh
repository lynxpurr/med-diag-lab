#!/bin/bash
# MED-A A3：五档 × 三集推理矩阵（按 GPU 现场空闲显存排程）
# 幂等：产物已存在即跳过；GPU free < 需求即挂"待 GPU"跳过——重跑本脚本补收。
# 用法：bash tools/run_med_a_matrix.sh   （console 全量由调用方 tee 收口）
set -u
cd "$(dirname "$0")/.."
: "${MODELS_DIR:?请先 export MODELS_DIR=<Qwen3 模型根目录>}"

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# GPU 采样（30s 周期，run 证据只增不改）
SAMPLER_LOG=output/med-a/gpu-sample.log
( while :; do echo "$(date +%H:%M:%S) free=$(gpu_free)MiB"; sleep 30; done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

# tag|模型|quant_bits(空=bf16)|batch|所需freeMiB —— 小档先行，大档 4bit 贴边纪律
# 所需freeMiB：2026-11-27 实测显存净增峰值（整卡 used 减常驻基线）加 200MiB、取整到百位——
#   0.6b 6485 / 1.7b 9673 / 4b 12047 / 8b 10583 / 14b 13187
PLAN=(
  "0.6b|$MODELS_DIR/Qwen3-0.6B||16|6700"
  "1.7b|$MODELS_DIR/Qwen3-1.7B||16|9900"
  "4b|$MODELS_DIR/Qwen3-4B-Instruct-2507||8|12300"
  "8b|$MODELS_DIR/Qwen3-8B|4|8|10800"
  "14b|$MODELS_DIR/Qwen3-14B|4|4|13400"
)

for row in "${PLAN[@]}"; do
  IFS='|' read -r tag model quant batch need <<< "$row"
  for s in val-reg test-ood test-rob; do
    out=output/med-a/${tag}-${s}.jsonl
    if [ -f "$out" ]; then echo "[skip] $tag/$s 已在"; continue; fi
    free=$(gpu_free)
    if [ "$free" -lt "$need" ]; then
      echo "[待GPU] $tag/$s 需 ${need}MiB 现仅 ${free}MiB —— 跳过，稍后重跑本脚本补收"
      continue
    fi
    echo "=== $tag/$s batch=$batch quant=${quant:-bf16} free=${free}MiB $(date +%H:%M:%S) ==="
    if [ -n "$quant" ]; then
      python3 tools/run_eval_med.py infer "$tag" "$model" "$s" --quant_bits "$quant" --batch "$batch" \
        || { echo "[fail] infer $tag/$s（实录保留，继续下一列）"; continue; }
    else
      python3 tools/run_eval_med.py infer "$tag" "$model" "$s" --batch "$batch" \
        || { echo "[fail] infer $tag/$s（实录保留，继续下一列）"; continue; }
    fi
    python3 tools/run_eval_med.py score "$tag" "$s" || echo "[fail] score $tag/$s"
  done
done
echo "=== 矩阵驱动结束 $(date +%H:%M:%S) ==="
ls -la output/med-a/*.jsonl 2>/dev/null | grep -v mini
