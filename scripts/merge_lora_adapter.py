"""Merge a Qwen3-VL LoRA adapter into its base model without overwriting output."""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
parser.add_argument("--adapter", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

adapter = Path(args.adapter)
destination = Path(args.output)
temporary = destination.with_name(".tmp_" + destination.name)
if destination.exists() or temporary.exists():
    raise SystemExit("STOP: output or temporary output already exists")
if not adapter.exists():
    raise SystemExit(f"STOP: adapter does not exist: {adapter}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.base, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
processor = AutoProcessor.from_pretrained(adapter)
temporary.parent.mkdir(parents=True, exist_ok=True)
model.save_pretrained(temporary, safe_serialization=True)
processor.save_pretrained(temporary)
os.replace(temporary, destination)
required = {"config.json", "model.safetensors", "processor_config.json"}
missing = sorted(name for name in required if not (destination / name).exists())
if missing:
    raise SystemExit(f"Merge verification failed; missing: {missing}")
print("Merged model:", destination)
