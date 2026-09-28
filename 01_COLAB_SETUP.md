# Colab setup

```bash
pip install -U transformers accelerate peft qwen-vl-utils datasets pillow requests pandas tqdm sentencepiece
```

Check GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

Run:

```bash
python vlm_kd_end_to_end.py prepare_train
python vlm_kd_end_to_end.py download_images
python vlm_kd_end_to_end.py validate_train

# Put the official MMBench Dev EN TSV at:
# data/eval_1000/MMBench_DEV_EN.tsv

python vlm_kd_end_to_end.py prepare_eval
python vlm_kd_end_to_end.py eval --model Qwen/Qwen3-VL-2B-Instruct --out results/baseline_2b
python vlm_kd_end_to_end.py eval --model Qwen/Qwen3-VL-8B-Instruct --out results/baseline_8b
python vlm_kd_end_to_end.py train_kd
python vlm_kd_end_to_end.py eval --model Qwen/Qwen3-VL-2B-Instruct --adapter checkpoints/online_kd_2b/final --out results/kd
python vlm_kd_end_to_end.py summary
```
