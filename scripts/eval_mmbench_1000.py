"""Evaluate a base Qwen3-VL-2B model plus an optional LoRA adapter."""

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data/eval_1000/MMBench_DEV_EN_1000.tsv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


args = parse_args()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map=args.device
)
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
processor = AutoProcessor.from_pretrained(args.model)
model.eval()
data = pd.read_csv(args.data, sep="\t")
torch.cuda.reset_peak_memory_stats()
results, correct, total_latency = [], 0, 0.0

for row_number, row in data.iterrows():
    image = Image.open(io.BytesIO(base64.b64decode(str(row["image"])))).convert("RGB")
    options = [f"{letter}. {row[letter]}" for letter in "ABCD" if pd.notna(row[letter])]
    hint = f"\nHint: {row['hint']}\n" if pd.notna(row["hint"]) and str(row["hint"]).strip() else ""
    prompt = (
        "Look at the image and answer the multiple-choice question.\n"
        "Return ONLY the letter of the correct option (A, B, C, or D).\n"
        f"{hint}\nQuestion: {row['question']}\n" + "\n".join(options)
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    match = re.search(r"\b([ABCD])\b", raw.upper())
    prediction = match.group(1) if match else ""
    expected = str(row["answer"]).strip().upper()
    is_correct = prediction == expected
    correct += int(is_correct)
    total_latency += latency
    results.append({
        "row_number": int(row_number), "index": int(row["index"]),
        "question": str(row["question"]), "expected": expected,
        "prediction": prediction, "raw_output": raw, "correct": is_correct,
        "latency_sec": latency,
    })
    if (row_number + 1) % 25 == 0:
        print(f"[{row_number + 1:4d}/{len(data)}] accuracy={correct / (row_number + 1) * 100:.2f}%")
    image.close()

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for result in results:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
summary = {
    "model": args.model, "adapter": args.adapter, "method": args.method,
    "num_samples": len(data), "correct": correct,
    "accuracy_percent": correct / len(data) * 100,
    "average_generation_latency_sec": total_latency / len(data),
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    "dtype": "float16", "decoding": "greedy", "max_new_tokens": 8,
    "evaluation_file": args.data,
}
summary_path = output.with_name(output.stem + "_summary.json")
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
