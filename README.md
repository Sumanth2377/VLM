# VLM Knowledge Distillation

## Current project setup

- Teacher: Qwen/Qwen3-VL-8B-Instruct
- Student: Qwen/Qwen3-VL-2B-Instruct
- FP16
- LoRA r=8, alpha=16, dropout=0.05
- LoRA targets: q_proj, k_proj, v_proj, o_proj
- 10,000 training records and 9,639 unique images
- 3 x NVIDIA RTX 6000 Ada, 48 GB each
- Training on GPU 0

## Training completed

| Method | Average loss | Final loss | Time | Peak VRAM |
|---|---:|---:|---:|---:|
| Online hard KD | 0.685716 | 0.558888 | 405 min* | 21.43 GB |
| Offline teacher-only KD | 0.685636 | 0.556590 | 14.55 min | 6.61 GB |
| Offline mixed KD | 1.136519 | 2.011141 | 18.27 min | 7.69 GB |

\*The online wall time includes a long pause.

The mixed offline objective uses 75% ground-truth targets and 25%
cached teacher-response targets with seed 123 and learning rate 1e-5.

## Official MMBench results

| Benchmark | Baseline | Teacher-only offline | Mixed offline |
|---|---:|---:|---:|
| MMBench DEV EN | 66.58% | 57.47% | 74.31% |
| MMBench DEV CN | 66.15% | - | 73.97% |

Both official evaluations used 4,329 circular-expanded samples with
exact-match scoring and had zero inference or judge failures.

## Repository scripts

- `vlm_kd_end_to_end.py`: original 1K online KD pipeline.
- `vlm_kd_online_10k.py`: expanded 10K online hard-response KD pipeline.

The offline, mixed-KD, merge, and VLMEvalKit scripts currently remain on the
Linux training machine and must be copied into this repository for a complete
source-code archive.

## Original 1K experiment

Teacher: Qwen/Qwen3-VL-8B-Instruct
Student: Qwen/Qwen3-VL-2B-Instruct
KD: online response-based KD
Precision: FP16
LoRA: r=8, alpha=16, dropout=0.05
Targets: q_proj, k_proj, v_proj, o_proj

Training:
- 1,000 records, seed 123
- 1,000 optimization steps
- learning rate 2e-5
- teacher max_new_tokens 64
- checkpoint every 100 steps
- final LoRA adapter about 24 MB
- 3,211,264 trainable parameters

Evaluation:
- fixed 1,000 MMBench Dev EN image-bearing questions
- evaluation subset seed 42
- greedy decoding
- max_new_tokens 8
- same evaluation set for teacher, baseline student and KD student

Measured:
- 8B: 90.00%, 0.0855 s, 16.49 GB inference VRAM
- 2B baseline: 80.40%, 0.0400 s, 4.07 GB inference VRAM
- 2B + KD: 81.90%, 0.0489 s, 4.08 GB inference VRAM
- KD training: 27.19 min, 21.41 GB peak VRAM
- loss: 0.9762 -> 0.5617
- average loss: 0.9027
