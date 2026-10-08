# Qwen3-VL Knowledge Distillation (VLM-KD)

**Offline response-level knowledge distillation** — Qwen3-VL-8B teacher → Qwen3-VL-2B student, trained on LLaVA-Instruct data, evaluated on MMBench DEV EN/CN.

---

## Overview

Transfer capabilities of **Qwen3-VL-8B-Instruct** to **Qwen3-VL-2B-Instruct** without increasing model size, using response-level KD with LoRA fine-tuning.

> **Correct terminology:** This is *offline response-level KD with ground-truth mixing* — NOT online KD, NOT simultaneous online+offline KD.

---

## Models

| Role | Model | Status |
|------|-------|--------|
| Teacher | `Qwen/Qwen3-VL-8B-Instruct` | Frozen (offline) |
| Student | `Qwen/Qwen3-VL-2B-Instruct` | LoRA fine-tuned |

Precision: **FP16** — only LoRA adapters are updated, base weights frozen.

---

## Dataset

Source: **LLaVA-Instruct 150K** (157,712 records)

| Scale | Training Records | Unique Images |
|-------|-----------------|---------------|
| 10K | 10,000 | 9,639 |
| 50K | 50,000 | 41,414 |

50K built by preserving all 10K records + 40K additional records (seed 123).

**Key fix:** Cache keyed by `(image, question)` — not record ID — since the same image can have multiple questions.

---

## Method

### 75/25 Mixed Supervision Schedule

| Experiment | Ground-truth steps | Teacher steps |
|------------|-------------------|---------------|
| 10K | 7,500 | 2,500 |
| 50K | 37,500 | 12,500 |

Each optimization step uses **one target** (not two simultaneous losses). Fixed shuffled schedule with seed 123.

Conceptually: `L = 0.75 * L_GT + 0.25 * L_teacher`

### Loss Function

- Token-level **Cross-Entropy** on assistant response tokens only
- Prompt and image tokens masked with `-100`
- No KL-divergence or logit distillation

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank r | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Seed | 123 |

---

## Teacher Response Generation

The 8B teacher generates responses **before** student training (offline) and stores them in JSONL files. The teacher is **not loaded** during student training.

- Generation: `do_sample=False`, `max_new_tokens=64`
- Cache key: `(image, question)` tuple
- 50K schedule: 12,500 teacher-target steps — all validated (0 missing)

---

## Results (Verified — MMBench via VLMEvalKit)

| Model | MMBench EN | MMBench CN |
|-------|-----------|-----------|
| Qwen3-VL-2B Baseline | 66.58% | 66.15% |
| **2B — 10K Offline Mixed KD** | **74.31%** | **73.97%** |
| Qwen3-VL-8B Teacher | 74.48% | 77.66% |

**10K KD gains over baseline: +7.73 pp EN, +7.82 pp CN**

The 10K KD model reaches near-parity with the 8B teacher on English (74.31% vs 74.48%).

> ⚠️ The old 50K result (65.12% EN / 64.09% CN) is **invalid** due to a corrupted teacher-response cache. The corrected 50K run is in progress.

---

## 50K Debugging Note

The first 50K run failed because cached teacher responses were mismatched with image/question pairs.

**Root cause:** Indexing by record ID alone (unsafe when one image has multiple questions).

**Fix:** Cache keyed by `(image, question)` tuple. All 12,500 teacher-target steps re-validated successfully.

---

## 50K Status

| Step | Status |
|------|--------|
| Teacher response generation (corrected) | ✅ Complete |
| Cache validation (12,500 / 12,500) | ✅ Passed |
| Student training | 🔄 In progress (step 3,000 / 50,000) |
| MMBench DEV EN (50K) | ⏳ Pending |
| MMBench DEV CN (50K) | ⏳ Pending |

Latest checkpoint: `checkpoints/offline_mixed_kd_2b_50k_final/step_3000`

---

## Experiments Summary

| Experiment | Notes |
|------------|-------|
| Online hard KD (10K) | Teacher in-loop during training |
| Offline teacher-only KD (10K) | 100% teacher targets |
| **Offline mixed KD 75/25 (10K)** | **Best result: 74.31% EN / 73.97% CN** ✅ |
| Ground-truth-only SFT (10K) | No teacher supervision |
| Offline mixed KD 75/25 (50K) | Corrected run in progress |

---

## Repository Contents

| File | Description |
|------|-------------|
| `vlm_kd_end_to_end.py` | End-to-end KD training script |
| `Vlm_KD.py` | Core KD implementation |
| `Qwen3VL_KD_50k.txt` | Complete 50K code reference and docs |
| `01_COLAB_SETUP.md` | Colab setup guide |
| `scripts/` | Offline KD, eval, and LoRA merge scripts |

---

## Acknowledgements

- Dataset: [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- Models: [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl-6796ffcebb4571b8b38afa5d)
- Evaluation: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- Images: [COCO train2017](https://cocodataset.org/)
