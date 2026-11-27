#!/bin/bash
# MED-B SFT run 脚本：两级（1.7B 快筛 / 14B 交付）× 三臂（f1/f2/f3c）——通用训练骨架的固化
# 2026-09-28 现场现实内嵌：14B 默认 bs1×accum16 + expandable_segments（free ~11.5GB，
#   13.4GB 锚是 1024 时代实测、2048 激活更高，bs2×accum8 仅在 free ≥13GB 时用）。
#   同日 OOM 实测修正：bs1@2048 在常驻 4.2GiB、总 free 11.2GiB 下首步触顶（进程 11.20GiB
#   差 42MiB）——NEED 11200→12300，对齐 skill"常驻之外 <12GiB 不上 14B 训练档"红线。
# 用法：
#   bash tools/run_med_b.sh 1.7b f1     # B2 快筛（唯一档）
#   bash tools/run_med_b.sh 14b f1      # B3 臂（f1/f2/f3c 各一 run，同 seed 同超参）
# 门：GPU free < 阈值拒跑（训练勿静默 OOM）；output 已存在即跳过（证据只增不改）。
# 跑完核对 args.json：max_length=2048 生效 ∧ loss_scale 实效口径（qwen3 自动升级
#   default+ignore_empty_think）∧ quant bnb-4 —— B3 门。
set -euo pipefail
cd "$(dirname "$0")/.."
SWIFT=${SWIFT_BIN:-swift}
: "${MODELS_DIR:?请先 export MODELS_DIR=<Qwen3 模型根目录>}"

TIER=${1:?用法: run_med_b.sh <1.7b|14b> <f1|f2|f3c>}
ARM=${2:?用法: run_med_b.sh <1.7b|14b> <f1|f2|f3c>}

case "$TIER" in
  1.7b) MODEL=$MODELS_DIR/Qwen3-1.7B; BS=2; ACCUM=8; NEED=7000; ALLOC="";;
  14b)  MODEL=$MODELS_DIR/Qwen3-14B; BS=1; ACCUM=16; NEED=12300; ALLOC="expandable_segments:True";;
  *) echo "未知档位 $TIER（微调只走两级：1.7b 快筛 / 14b 交付）"; exit 2;;
esac
case "$ARM" in
  f1|f2|f3c) DATASET=data/med0/swift-train-${ARM}.jsonl;;
  *) echo "未知臂 $ARM"; exit 2;;
esac

OUT=output/med-b/${TIER}-${ARM}
if [ -e "$OUT" ]; then echo "[skip] $OUT 已在（证据只增不改）"; exit 0; fi

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt "$NEED" ]; then
  echo "[拒跑] $TIER/$ARM 需 ${NEED}MiB 现仅 ${FREE}MiB——训练勿静默 OOM，等释放后重跑本脚本"
  exit 1
fi
echo "=== $TIER/$ARM model=$MODEL bs=$BS×accum=$ACCUM free=${FREE}MiB $(date +%F\ %T) ==="

mkdir -p "$OUT"
( while :; do echo "$(date +%T) free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)MiB"; sleep 30; done ) > "$OUT/gpu-sample.log" &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

PYTORCH_CUDA_ALLOC_CONF=$ALLOC $SWIFT sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --template qwen3 --enable_thinking false \
  --tuner_type lora --tuner_backend peft --lora_rank 16 --lora_alpha 32 --lora_dropout 0 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 \
  --num_train_epochs 3 --per_device_train_batch_size $BS --gradient_accumulation_steps $ACCUM \
  --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --max_length 2048 --bf16 true --gradient_checkpointing true --packing false \
  --loss_scale default --logging_steps 1 --save_strategy epoch --save_only_model true \
  --seed 20260925 --data_seed 20260925 --report_to none --add_version false \
  --output_dir "$OUT"

echo "[done] $TIER/$ARM → $OUT"
echo "[门] 核对快照：python3 -c \"import json; a=json.load(open('$OUT/args.json')); print({k:a[k] for k in ('max_length','loss_scale','quant_method','quant_bits','per_device_train_batch_size','gradient_accumulation_steps','seed')})\""
echo "[门] 末步 ckpt：ls $OUT/checkpoint-* | tail -1（评测取末步不写死）"
