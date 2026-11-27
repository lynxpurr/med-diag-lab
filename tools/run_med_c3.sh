#!/bin/bash
# MED-C C4 评测一键脚本：三臂 × 五集十五连 + 主表。
# 前置：output/med-c/14b-{cg,cm,cn}/checkpoint-225 齐（三臂增强重训完成）。
# 幂等：score 产物已存在即跳过（证据只增不改）。
# 末步硬门 225（1190 行 → 75 步/ep × 3；sort -V 数值感知，路径连字符教训见 lessons MED-B 执行批）。
set -euo pipefail
cd "$(dirname "$0")/.."
OUTC=output/med-c

echo "=== [C4] 三臂 × 五集评测 $(date +%T) ==="
for arm in cg cm cn; do
  CKPT=$(ls -d $OUTC/14b-$arm/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$CKPT" ] || { echo "[拒] 14b-$arm 无 checkpoint（三臂未完成？）"; exit 1; }
  case "$CKPT" in
    *checkpoint-225) : ;;  # 1190/16=74.375→75 步/ep ×3ep；非 225 即选择器坏，硬门
    *) echo "[拒] $arm 选中非末步 ckpt：$CKPT（应为 checkpoint-225）"; exit 1 ;;
  esac
  echo "-- $arm 末步 ckpt: $CKPT"
  for s in val-reg test-ood test-rob neg negn; do
    if [ -e "$OUTC/score-sft-medc-$arm-$s.json" ]; then echo "[skip] sft-medc-$arm/$s"; continue; fi
    python3 tools/run_eval_med.py infer sft-medc-$arm $s --adapters "$CKPT" --out $OUTC
    python3 tools/run_eval_med.py score sft-medc-$arm $s --out $OUTC
  done
done

echo "=== [C4] 主表 ==="
python3 tools/run_eval_med.py table --out $OUTC | tee $OUTC/c4-table.txt

echo "=== [C4] done $(date +%T) —— R5'/R6 判卷提示 ==="
echo "R5'（PLAN §4 MED-C 校准块）：三条件保持（garbled/minimal ≥20/20 冠军锚；无病 neg ≥19/20 且 neg-013 不恶化）"
echo "∧ 主判据 val-reg F1 ∈ 0.800±2 分等价带 ∧ 其余轴零漂移（格式 100%/覆盖 ≥0.937/幻觉 0/ood 不劣化超 3 分）"
echo "对照锚（MED-B 冠军）：F1 0.800 · 格式 80/80 · 覆盖 0.976 · 幻觉 0 · ood 0.133 · garbled 20/20 · minimal 20/20 · neg 19/20 · negn 10/10"
