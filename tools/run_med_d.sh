#!/bin/bash
# MED-D DPO run 脚本：步数阶梯三臂（s1/s3/s9）× 挂 MED-B f1 冠军双 adapters
# 谱系：上游方法学仓的 run_t4.sh（DPO 全套注记）+ run_med_b.sh（GPU 预检门/采样 loop/幂等跳过）
# 设计（PLAN §4 MED-D + playbook §3）：
#   --adapters 与 --ref_adapters 同指 SFT 冠军 output/med-b/14b-f1/checkpoint-219
#   （同一 4bit base 挂冻结 ref，显存不翻倍，16GB 可行——rlhf.py L178-191 路径）；
#   lr 5e-6（SFT 谱系 2e-4 的 1/40）；β 0.1 显式（不扫——C-1 结论平移：β 不敏感）；
#   步数是开关型：48 对×3 步已压满行为轴，9 步过校正 → 阶梯 1/3/9 用 --max_steps 控制；
#   save_strategy steps+save_steps=N 保 ckpt 恰落在臂步数（300 对 bs1×accum16 → 19 步/epoch，
#   臂均远小于 1 epoch，属刻意的亚 epoch 剂量）。
# DPO 专属显存适配（2026-09-29 s1 首步 OOM 实测后加，实录 14b-dpo-s1-oom-0929）：
#   DPO 比 SFT 多 ref 前向 + chosen/rejected 双序列（bs1 实为 2 seq）→ 12.60GiB 差 98MiB。
#   ① --optim adamw_bnb_8bit：LoRA 优化器状态 fp32→int8 省 ~0.9GB（1-9 步 5e-6 剂量数值无感，
#      与 SFT 谱系的已知配置差，报告记账）——注意其状态在首个 optimizer.step 才物化，救不了
#      首个 backward 峰值，作用在 s9 臂的稳态；② --max_length 1360：check_dpo_med 实测 300 对
#   encode 最长 1613（含全部模板 token）——SFT 的 2048 线为 F1 全链 1725 设，DPO 对是子集，
#   1360 > 实测 1342 保零截断（探针 1400 口径内）；③ --torch_empty_cache_steps 1：micro-batch 间清瞬态碎片
#   （OOM 实测缺口仅 ~30-300MiB = bnb 反向去量化缓冲 × 最长对的边界激活）。
# 数据：data/medd/dpo-pairs.jsonl（300 对，check_dpo_med.py 对拍全绿是前置）。
# 用法：bash tools/run_med_d.sh <s1|s3|s9>
# 门：GPU free < 12900 拒跑——DPO 实测峰值 12.39GiB（s3 logging）只对"仅常驻服务"、free 12976 的口径成立；第三方进程 610MiB 上卡即差 ~7MiB 触顶（s9 实录），等窗口再跑；
#     output 已存在即跳过（证据只增不改）；跑完核对 args.json（rlhf_type/lr/β/ref_adapters）。
set -euo pipefail
cd "$(dirname "$0")/.."
SWIFT=${SWIFT_BIN:-swift}
: "${MODELS_DIR:?请先 export MODELS_DIR=<Qwen3 模型根目录>}"

ARM=${1:?用法: run_med_d.sh <s1|s3|s9>}
case "$ARM" in
  s1) STEPS=1;; s3) STEPS=3;; s9) STEPS=9;;
  *) echo "未知臂 $ARM（步数阶梯：s1/s3/s9）"; exit 2;;
esac

MODEL=$MODELS_DIR/Qwen3-14B
CHAMP=output/med-b/14b-f1/checkpoint-219
DATASET=data/medd/dpo-pairs.jsonl
OUT=output/med-d/14b-dpo-${ARM}
NEED=12900

test -f "$CHAMP/args.json" || { echo "[拒跑] 冠军挂点缺 args.json：$CHAMP"; exit 1; }
if [ -e "$OUT/checkpoint-$STEPS" ]; then echo "[skip] $OUT/checkpoint-$STEPS 已在（证据只增不改）"; exit 0; fi

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt "$NEED" ]; then
  echo "[拒跑] med-d/$ARM 需 ${NEED}MiB 现仅 ${FREE}MiB——训练勿静默 OOM，等释放后重跑本脚本"
  exit 1
fi
echo "=== med-d/$ARM steps=$STEPS 双adapters=$CHAMP free=${FREE}MiB $(date +%F\ %T) ==="

mkdir -p "$OUT"
( while :; do echo "$(date +%T) free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)MiB"; sleep 30; done ) > "$OUT/gpu-sample.log" &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $SWIFT rlhf \
  --rlhf_type dpo \
  --model "$MODEL" \
  --adapters "$CHAMP" \
  --ref_adapters "$CHAMP" \
  --dataset "$DATASET" \
  --template qwen3 --enable_thinking false \
  --tuner_type lora --tuner_backend peft --lora_rank 16 --lora_alpha 32 --lora_dropout 0 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 --beta 0.1 --loss_type sigmoid \
  --optim adamw_bnb_8bit --torch_empty_cache_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --max_length 1360 --bf16 true --gradient_checkpointing true --packing false \
  --logging_steps 1 --save_strategy steps --save_steps "$STEPS" --save_only_model true \
  --seed 20260925 --data_seed 20260925 --report_to none --add_version false \
  --output_dir "$OUT" 2>&1 | tee "$OUT/console.log"

echo "[done] med-d/$ARM → $OUT"
echo "[门] 核对快照：python3 -c \"import json; a=json.load(open('$OUT/args.json')); print({k:a.get(k) for k in ('rlhf_type','learning_rate','beta','loss_type','adapters','ref_adapters','max_steps','max_length','quant_method','quant_bits','seed')})\""
echo "[门] 末步 ckpt：ls $OUT/checkpoint-* | tail -1（应恰为 checkpoint-$STEPS）"
