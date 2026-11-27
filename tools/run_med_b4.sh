#!/bin/bash
# MED-B B4 评测一键脚本：三臂 × 三集九连 + 主表 + 零样本反面探针（30 条无病）
# 前置：output/med-b/14b-{f1,f2,f3c}/checkpoint-* 齐（三臂训练完成）
# 幂等：score 产物已存在即跳过该步（证据只增不改）；infer 产物在而 score 缺 → 脚本会因
#   infer 的 assert 停（人工处理），避免静默重跑覆盖证据。
# 冠军探针（--adapters 冠军臂 ckpt）不在此脚本——R2 判决出冠军后单跑，命令在尾注。
set -euo pipefail
cd "$(dirname "$0")/.."
OUTB=output/med-b
: "${MODELS_DIR:?请先 export MODELS_DIR=<Qwen3 模型根目录>}"
M14=$MODELS_DIR/Qwen3-14B

echo "=== [B4-1] 三臂九连评测 $(date +%T) ==="
for arm in f1 f2 f3c; do
  # sort -V（数值感知）：普通 sort 被路径内连字符干扰——checkpoint-73 会排到 219 后（本日实录）
  CKPT=$(ls -d $OUTB/14b-$arm/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$CKPT" ] || { echo "[拒] 14b-$arm 无 checkpoint（三臂未完成？）"; exit 1; }
  case "$CKPT" in
    *checkpoint-219) : ;;  # 3ep×1154/16≈217 → 末步必为 219；非 219 即选择器又坏了，硬门
    *) echo "[拒] $arm 选中非末步 ckpt：$CKPT（应为 checkpoint-219）"; exit 1 ;;
  esac
  echo "-- $arm 末步 ckpt: $CKPT"
  for s in val-reg test-ood test-rob; do
    if [ -e "$OUTB/score-sft-14b-$arm-$s.json" ]; then echo "[skip] sft-14b-$arm/$s"; continue; fi
    python3 tools/run_eval_med.py infer sft-14b-$arm $s --adapters "$CKPT" --out $OUTB
    python3 tools/run_eval_med.py score sft-14b-$arm $s --out $OUTB
  done
done

echo "=== [B4-2] 主表 ==="
python3 tools/run_eval_med.py table --out $OUTB | tee $OUTB/b4-table.txt

echo "=== [B4-3] 零样本反面探针（14B base × 30 条无病）==="
for s in neg-draft negn-draft; do
  if [ -e "$OUTB/score-zs-14b-$s.json" ]; then echo "[skip] zs-14b/$s"; continue; fi
  python3 tools/run_eval_med.py infer zs-14b "$M14" $s --quant_bits 4 --batch 1 --out $OUTB
  python3 tools/run_eval_med.py score zs-14b $s --out $OUTB
done

echo "=== [B4] done $(date +%T) —— 判据落卷提示 ==="
echo "R2/R3/R4 按 PLAN §4 原文判（主判据 val-reg F1；臂间差<2 分等价带先 seed 复核）"
echo "冠军探针（判决后单跑）：python3 tools/run_eval_med.py infer sft-14b-<冠军臂> neg-draft \\"
echo "  --adapters $OUTB/14b-<冠军臂>/checkpoint-<末步> --out $OUTB && python3 tools/run_eval_med.py score sft-14b-<冠军臂> neg-draft --out $OUTB"
