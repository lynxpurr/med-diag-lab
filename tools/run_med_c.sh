#!/bin/bash
# MED-C 增强重训 run 脚本：三臂（cg 乱码 / cm 极简 / cn 无病）× 14B，挂 MED-B 冠军
# ckpt-219 追加 LoRA（R6：不重训底座——--adapters 续训，run_t4.sh 同机制先例）。
# 数据 = data/medc/swift-train-c{g,m,n}.jsonl（f1 1154 原序 + aug 36，LAB-E 臂合并约定），
# 超参与 MED-B 全同（bs1×accum16 @2048 · lr 2e-4 · 3ep · seed 20260925）——臂间唯一变量 = 增强形态。
# 末步硬门：1190 行 → 75 步/ep × 3 = ckpt-225（run_med_b4.sh 的 219 硬门是 1154 行时代的值，
# 步数随数据集大小变——此处置门 225，选择器坏则大声失败）。
# 用法：bash tools/run_med_c.sh [cg|cm|cn|all]    # all = 串行三臂；output 已在即跳过（幂等）
# 门：GPU free < 12300MiB 拒跑（OOM 实测背书，B3 0928 实录）；跑完核对 args.json（adapters/超参）。
set -euo pipefail
cd "$(dirname "$0")/.."
SWIFT=${SWIFT_BIN:-swift}
: "${MODELS_DIR:?请先 export MODELS_DIR=<Qwen3 模型根目录>}"
MODEL=$MODELS_DIR/Qwen3-14B
CHAMP=output/med-b/14b-f1/checkpoint-219
NEED=12300
LAST_STEP=225

ARM=${1:-all}
[ -d "$CHAMP" ] || { echo "[拒] 冠军 ckpt 不在：$CHAMP"; exit 1; }

run_arm() {
  local ARM=$1 DS=data/medc/swift-train-${1}.jsonl
  local OUT=output/med-c/14b-${ARM}
  if [ -e "$OUT" ]; then echo "[skip] $OUT 已在（证据只增不改）"; return 0; fi
  # 末步硬门随数据集大小变（1190→225 / 1166→219）：步数 = ceil(行/16) × 3ep
  local ROWS=$(wc -l < "$DS")
  local LAST_STEP=$(( (ROWS + 15) / 16 * 3 ))
  local FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -lt "$NEED" ]; then
    echo "[拒跑] $ARM 需 ${NEED}MiB 现仅 ${FREE}MiB——等释放后重跑（勿贴边硬上，0928 OOM 实录）"
    return 1
  fi
  echo "=== med-c/$ARM dataset=$DS free=${FREE}MiB 挂冠军 $CHAMP $(date +%F\ %T) ==="
  mkdir -p "$OUT"
  ( while :; do echo "$(date +%T) free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)MiB"; sleep 30; done ) > "$OUT/gpu-sample.log" &
  local SAMPLER=$!
  trap "kill $SAMPLER 2>/dev/null" EXIT

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $SWIFT sft \
    --model "$MODEL" \
    --adapters "$CHAMP" \
    --dataset "$DS" \
    --template qwen3 --enable_thinking false \
    --tuner_type lora --tuner_backend peft --lora_rank 16 --lora_alpha 32 --lora_dropout 0 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 \
    --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --max_length 2048 --bf16 true --gradient_checkpointing true --packing false \
    --loss_scale default --logging_steps 1 --save_strategy epoch --save_only_model true \
    --seed 20260925 --data_seed 20260925 --report_to none --add_version false \
    --output_dir "$OUT"

  kill $SAMPLER 2>/dev/null || true
  # args.json 门：adapters 续训生效 ∧ MED-B 同款超参
  python3 - "$OUT/args.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
keys = ("adapters", "max_length", "loss_scale", "quant_method", "quant_bits",
        "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "seed")
print({k: a.get(k) for k in keys})
assert a.get("max_length") == 2048 and a.get("seed") == 20260925
assert a.get("adapters"), "adapters 未落 args.json（续训没生效？）"
PY
  echo "[done] $ARM → $OUT（末步应为 checkpoint-$LAST_STEP，行数 $ROWS）"
}

case "$ARM" in
  cg|cm|cn|cn-low) run_arm "$ARM" ;;
  all) for a in cg cm cn; do run_arm "$a"; done ;;
  *) echo "用法: run_med_c.sh <cg|cm|cn|cn-low|all>"; exit 2 ;;
esac
