# Qwen3-VL Knowledge Distillation (VLM-KD)

> **Offline response-level knowledge distillation** with ground-truth mixing —  
> Qwen3-VL-8B teacher -> Qwen3-VL-2B student, trained on LLaVA-Instruct data, evaluated on MMBench DEV EN/CN.

---

## Overview

The primary objective of this project is to investigate whether the capabilities of the larger **Qwen3-VL-8B-Instruct** model can be transferred to the smaller **Qwen3-VL-2B-Instruct** student without increasing the student model size.

We focus on:
- **Response-level** knowledge distillation (not logit/KL-divergence distillation)
- **LoRA-based** parameter-efficient fine-tuning
- **Offline** teacher response generation (teacher not loaded during student training)
- **75/25 ground-truth / teacher-response mixing** schedule
- Official **MMBench DEV EN & CN** evaluation via VLMEvalKit

> **Correct terminology:** This is *offline response-level KD with ground-truth mixing*. It is **not** online KD and does **not** combine online and offline KD simultaneously.

---

## Models

| Role    | Model                         | Status           |
|---------|-------------------------------|------------------|
| Teacher | Qwen/Qwen3-VL-8B-Instruct   | Frozen (offline) |
| Student | Qwen/Qwen3-VL-2B-Instruct   | LoRA fine-tuned  |

Training precision: **FP16**. Student base weights remain frozen; only LoRA adapter parameters are updated.

---

## Dataset

Source: **LLaVA-Instruct 150K** (157,712 records)

| Scale | Training Records | Unique Images |
|-------|-----------------|---------------|
| 10K   | 10,000          | 9,639         |
| 50K   | 50,000          | 41,414        |

The 50K dataset was built by preserving all 10K records and sampling 40,000 additional records using random seed 123.

**Key insight:** The same image can appear with multiple different questions. Teacher response caching therefore uses (image, question) as the cache key to prevent cross-question response mismatches.

---

## Method

### Pipeline

`
LLaVA-Instruct data
        |
Prepare 10K / 50K training records
        |
Qwen3-VL-8B Teacher (frozen)
        |
Generate teacher responses offline
        |
Cache responses using (image, question)
        |
75% Ground Truth  +  25% Teacher Responses
        |
Qwen3-VL-2B Student + LoRA
        |
Token-level Cross-Entropy on assistant response tokens
        |
AdamW optimization
        |
Trained 2B LoRA adapter
        |
Official MMBench DEV EN + CN
        |
Compare against 2B baseline and 8B teacher
`

### 75/25 Mixed Supervision Schedule

A fixed, seed-reproducible shuffled schedule determines which target is used at each step:

| Experiment | Ground-truth steps | Teacher-response steps |
|------------|-------------------|----------------------|
| 10K        | 7,500             | 2,500                |
| 50K        | 37,500            | 12,500               |

Conceptually: L = 0.75 * L_GT + 0.25 * L_teacher

In practice, **each step uses exactly one target** (not two simultaneous losses). The weighted objective is the expected objective of the shuffled schedule.

### Loss Function

- **Loss:** Token-level Cross-Entropy
- **Input:** Image + Question
- **Target:** Ground-truth GPT response *or* offline Qwen3-VL-8B teacher response
- **Masking:** Prompt/image tokens masked with -100 — only assistant response tokens contribute to loss
- **Optimization:** loss.backward() -> AdamW -> LoRA update

No KL-divergence / logit distillation or intermediate visual-feature distillation is used.

### LoRA Configuration

| Parameter       | Value                                  |
|----------------|----------------------------------------|
| Rank r          | 8                                      |
| Alpha           | 16                                     |
| Dropout         | 0.05                                   |
| Target modules  | q_proj, k_proj, v_proj, o_proj         |
| Bias            | none                                   |
| Task type       | causal language modeling               |

### Training Configuration

| Parameter       | Value    |
|----------------|----------|
| Optimizer       | AdamW    |
| Learning rate   | 1e-5     |
| Precision       | FP16     |
| Random seed     | 123      |
| Device          | CUDA GPU |
| Scheduler       | none     |
| Warmup          | none     |

Checkpoint saving and resume support are included for the longer 50K run.

---

## Teacher Response Generation (Offline)

The 8B teacher generates responses **before** student training and stores them in JSONL files. During student training, the teacher model is **not** loaded.

- Teacher: Qwen3-VL-8B-Instruct
- Generation: do_sample=False, max_new_tokens=64 (greedy/deterministic)
- Cache format: {"image": "...", "question": "...", "teacher_response": "..."}
- 10K cache reused: 2,495 teacher responses carried over
- 50K total teacher-target steps: 12,500 — all validated (0 missing)

---

## Results (Verified)

Evaluated using **VLMEvalKit** on MMBench DEV EN and DEV CN:

| Model                         | MMBench EN | MMBench CN |
|-------------------------------|-----------|-----------|
| Qwen3-VL-2B Baseline          | 66.58%    | 66.15%    |
| **2B - 10K Offline Mixed KD** | **74.31%**| **73.97%**|
| Qwen3-VL-8B Teacher           | 74.48%    | 77.66%    |

**10K KD gains over 2B baseline:**
- EN: **+7.73 pp**
- CN: **+7.82 pp**

The 10K KD model achieves near-parity with the 8B teacher on English (74.31% vs 74.48%).

> **Note on 50K results:** The first 50K run produced 65.12% EN / 64.09% CN due to a corrupted/mismatched teacher-target cache. These numbers are **invalid** and should not be cited as final results. The corrected 50K run uses (image, question)-keyed caching and has been restarted from a checkpoint at step 3,000 / 50,000.

---

## Important Debugging Note - 50K Cache Issue

The first 50K run failed because some cached teacher responses were **mismatched** with their image/question pairs.

**Root cause:** Unsafe response indexing by record ID alone.

**Fix:** Cache keyed by (image, question) tuple. After fix, all 12,500 teacher-target steps were validated successfully.

---

## 50K Status

| Step | Status |
|------|--------|
| Teacher response generation (corrected) | Complete |
| Teacher cache validation (12,500 / 12,500) | Passed |
| Student training | In progress (resumed from step 3,000) |
| MMBench DEV EN (50K) | Pending |
| MMBench DEV CN (50K) | Pending |

Latest safe checkpoint: checkpoints/offline_mixed_kd_2b_50k_final/step_3000

---

## Experiments Summary

| Experiment                                | Notes                                           |
|-------------------------------------------|-------------------------------------------------|
| Online hard KD (10K)                      | Teacher in-loop during student training         |
| Offline teacher-only KD (10K)             | 100% teacher targets                            |
| **Offline mixed KD - 75/25 (10K)**        | **Best verified: 74.31% EN / 73.97% CN**        |
| Ground-truth-only SFT (10K)               | No teacher supervision                          |
| Offline mixed KD - 75/25 (50K, corrected) | In progress                                     |

---

## Relationship to LLaVA-KD

LLaVA-KD uses a broader multi-stage distillation framework involving response, visual-token, and relation-level distillation. This project deliberately implements a **narrower scope**: offline response-level KD + LoRA, focused on measuring the practical effect of ground-truth mixing at 10K and 50K scale.

---

## One-Sentence Technical Summary

"We freeze the Qwen3-VL-2B base model and train LoRA adapters using token-level cross-entropy on assistant response tokens, where 75% of optimization steps use the original ground-truth answer and 25% use responses generated offline by the frozen Qwen3-VL-8B teacher."

---

## Repository Contents

| File | Description |
|------|-------------|
| lm_kd_end_to_end.py | End-to-end VLM KD training script |
| Vlm_KD.py | Core VLM KD implementation |
| Qwen3VL_KD_50k.txt | Complete 50K project code reference and documentation |
|  1_COLAB_SETUP.md | Colab environment setup instructions |
| README.md | This file |

---

## Acknowledgements

- Dataset: [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- Teacher/Student models: [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl-6796ffcebb4571b8b38afa5d)
- Evaluation: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- Images: [COCO train2017](https://cocodataset.org/)
