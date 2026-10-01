# Recovered remote work

These source files were reconstructed from the remote terminal transcripts saved in the project chat because the Linux machine was unavailable during this update.

- `scripts/train_offline_kd_10k.py`: pure cached hard-response KD.
- `scripts/train_offline_mixed_kd_10k.py`: fixed-seed 75% ground-truth / 25% teacher stochastic mixture.
- `scripts/eval_mmbench_1000.py`: parameterized form of the two nearly identical 1,000-example evaluators.
- `scripts/merge_lora_adapter.py`: reusable form of the successful inline merge command.
- `configs/*.json`: exact VLMEvalKit configurations shown in the terminal output.

The mixed objective is expectation-equivalent to

`0.75 * CE(student, ground_truth) + 0.25 * CE(student, teacher_response)`.

It uses exactly 7,500 ground-truth targets and 2,500 cached-teacher targets over 10,000 steps. It is not simultaneous online/offline KD: the teacher answers were produced earlier and read from disk.

Verified remote outcomes:

- Pure offline KD, official MMBench DEV EN: 57.47%.
- Qwen3-VL-2B baseline, official MMBench DEV EN: 66.58%.
- Mixed offline KD, official MMBench DEV EN: 74.31%.
- Baseline, official MMBench DEV CN: 66.15%.
- Mixed offline KD, official MMBench DEV CN: 73.97%.
- Mixed training: 10,000 steps, 18.27 minutes, 7.69 GB peak VRAM.

Model weights, adapters, datasets, caches, logs, and generated results are intentionally not committed.
