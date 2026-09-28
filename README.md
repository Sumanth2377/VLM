# VLM KD — recovered reproduction package

Current project setup:
Experiment:
- Teacher: Qwen/Qwen3-VL-8B-Instruct
- Student: Qwen/Qwen3-VL-2B-Instruct
- Online response-based KD
- FP16
- LoRA r=8, alpha=16, dropout=0.05
- 1,000 training records
- 1,000 fixed MMBench Dev EN evaluation questions
- 1,000 optimization steps

# Project details

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
