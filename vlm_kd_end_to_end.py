"""
VLM Knowledge Distillation - 
====================================================

Teacher : Qwen/Qwen3-VL-8B-Instruct
Student : Qwen/Qwen3-VL-2B-Instruct
KD      : Online response-based KD
Precision: FP16
LoRA    : r=8, alpha=16, dropout=0.05 on q/k/v/o projections

Pipeline:
1. Prepare 1,000 LLaVA-Instruct training records (seed 123).
2. Download/validate referenced COCO train2017 images.
3. Prepare fixed 1,000-question MMBench Dev EN evaluation subset (seed 42).
4. Evaluate 8B teacher and 2B baseline.
5. Train 2B with online teacher-response CE KD + LoRA.
6. Evaluate the KD student on the SAME fixed 1,000 MMBench questions.
7. Print final comparison.

IMPORTANT:
- The original project used a fixed 1,000-record training subset and a separate
  fixed 1,000-question MMBench evaluation subset.
- The original project did NOT use k-fold cross-validation.
- The implemented KD loss is response-focused cross-entropy, not KL/logit KD
  and not intermediate-feature KD.
- The reported accuracies are project-subset results, not official MMBench scores.

Colab usage:
  pip install -U transformers accelerate peft qwen-vl-utils datasets pillow requests pandas tqdm
  python vlm_kd_end_to_end.py prepare_train
  python vlm_kd_end_to_end.py download_images
  python vlm_kd_end_to_end.py validate_train
  # Put official MMBench_DEV_EN.tsv at data/eval_1000/MMBench_DEV_EN.tsv
  python vlm_kd_end_to_end.py prepare_eval
  python vlm_kd_end_to_end.py eval --model Qwen/Qwen3-VL-2B-Instruct --out results/baseline_2b
  python vlm_kd_end_to_end.py eval --model Qwen/Qwen3-VL-8B-Instruct --out results/baseline_8b
  python vlm_kd_end_to_end.py train_kd
  python vlm_kd_end_to_end.py eval --model Qwen/Qwen3-VL-2B-Instruct --adapter checkpoints/online_kd_2b/final --out results/kd
  python vlm_kd_end_to_end.py summary
"""

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(".")
TRAIN_DIR = ROOT / "data/train_llava"
IMAGE_DIR = TRAIN_DIR / "images"
EVAL_DIR = ROOT / "data/eval_1000"
RESULTS_DIR = ROOT / "results"
KD_DIR = ROOT / "results/kd"
CKPT_DIR = ROOT / "checkpoints/online_kd_2b"

TRAIN_ANN = TRAIN_DIR / "llava_instruct_150k.json"
TRAIN_SUBSET = TRAIN_DIR / "train_1000.json"
IMAGE_LIST = TRAIN_DIR / "image_list.txt"
MMBENCH_SOURCE = EVAL_DIR / "MMBench_DEV_EN.tsv"
MMBENCH_FIXED = EVAL_DIR / "MMBench_DEV_EN_1000.tsv"

TEACHER = "Qwen/Qwen3-VL-8B-Instruct"
STUDENT = "Qwen/Qwen3-VL-2B-Instruct"

TRAIN_SEED = 123
EVAL_SEED = 42
N_TRAIN = 1000
N_EVAL = 1000
STEPS = 1000
LR = 2e-5
DTYPE = torch.float16

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def ensure_dirs():
    for p in [TRAIN_DIR, IMAGE_DIR, EVAL_DIR, RESULTS_DIR, KD_DIR, CKPT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def prepare_train():
    """Download LLaVA-Instruct-150K and create fixed 1,000-record subset."""
    ensure_dirs()
    url = (
        "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/"
        "playground/data/llava_instruct_150k.json"
    )
    if not TRAIN_ANN.exists():
        print("Downloading LLaVA annotations...")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        TRAIN_ANN.write_bytes(r.content)

    records = json.loads(TRAIN_ANN.read_text(encoding="utf-8"))
    print("Original records:", len(records))

    rng = random.Random(TRAIN_SEED)
    selected = rng.sample(records, N_TRAIN)
    TRAIN_SUBSET.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    names = sorted({r["image"] for r in selected if "image" in r})
    IMAGE_LIST.write_text("\n".join(names) + "\n", encoding="utf-8")

    print("Selected records:", len(selected))
    print("Unique images:", len(names))
    print("Saved:", TRAIN_SUBSET)


def download_images():
    """Download exactly the referenced COCO train2017 images."""
    ensure_dirs()
    names = [x.strip() for x in IMAGE_LIST.read_text().splitlines() if x.strip()]
    base = "http://images.cocodataset.org/train2017"

    def one(name):
        dst = IMAGE_DIR / name
        if dst.exists() and dst.stat().st_size > 0:
            return name, True, "exists"
        for attempt in range(3):
            try:
                r = requests.get(f"{base}/{name}", timeout=60, verify=True)
                r.raise_for_status()
                Image.open(BytesIO(r.content)).verify()
                dst.write_bytes(r.content)
                return name, True, "downloaded"
            except Exception as e:
                if attempt == 2:
                    return name, False, str(e)
        return name, False, "unknown"

    failed = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(one, n) for n in names]
        for f in tqdm(as_completed(futures), total=len(futures)):
            name, ok, msg = f.result()
            if not ok:
                failed.append((name, msg))

    print("Required images:", len(names))
    print("Failed:", len(failed))
    if failed:
        for x in failed[:20]:
            print("FAILED", x)


def validate_train():
    """Validate records and image readability."""
    records = json.loads(TRAIN_SUBSET.read_text(encoding="utf-8"))
    names = sorted({r["image"] for r in records if "image" in r})
    missing, corrupt = [], []
    for name in names:
        p = IMAGE_DIR / name
        if not p.exists():
            missing.append(name)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            corrupt.append(name)

    print("Training records:", len(records))
    print("Unique images:", len(names))
    print("Missing:", len(missing))
    print("Corrupt:", len(corrupt))
    assert len(records) == 1000
    assert not missing and not corrupt


def prepare_eval():
    """Create fixed 1,000 image-bearing MMBench Dev EN subset."""
    if not MMBENCH_SOURCE.exists():
        raise FileNotFoundError(
            f"Put the official MMBench Dev EN TSV at {MMBENCH_SOURCE}"
        )
    df = pd.read_csv(MMBENCH_SOURCE, sep="\t")
    image_df = df[df["image"].notna()].copy()
    fixed = image_df.sample(N_EVAL, random_state=EVAL_SEED)
    fixed.to_csv(MMBENCH_FIXED, sep="\t", index=False)
    print("Total rows:", len(df))
    print("Image rows:", len(image_df))
    print("Fixed evaluation rows:", len(fixed))
    print("Unique indices:", fixed["index"].nunique())
    print("Missing image fields:", fixed["image"].isna().sum())


def load_vl(model_name, adapter=None):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    print("Loading", model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def answer_letter(text):
    text = str(text).strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def mmbench_prompt(row):
    return (
        f"{row['question']}\n\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n\n"
        "Answer with only A, B, C, or D."
    )


def evaluate(model_name, adapter=None, out_dir="results/eval"):
    """Evaluate one model on the exact fixed 1,000 MMBench rows."""
    from transformers import AutoProcessor
    ensure_dirs()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MMBENCH_FIXED, sep="\t")
    model, processor = load_vl(model_name, adapter)
    device = next(model.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_path = Path(str(row["image"]))
        if not image_path.exists():
            # If the TSV stores just a filename, try the fixed local folder.
            image_path = EVAL_DIR / str(row["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Evaluation image not found: {row['image']}")

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": mmbench_prompt(row)},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        response = processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        pred = answer_letter(response)
        expected = str(row["answer"]).strip().upper()
        results.append({
            "index": int(row["index"]),
            "prediction": pred,
            "expected": expected,
            "correct": pred == expected,
            "response": response,
            "latency_sec": latency,
        })

    correct = sum(x["correct"] for x in results)
    accuracy = correct / len(results)
    avg_latency = sum(x["latency_sec"] for x in results) / len(results)
    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available() else 0.0
    )

    jsonl = out_dir / "mmbench_1000.jsonl"
    summary = out_dir / "mmbench_1000_summary.json"
    with jsonl.open("w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary.write_text(json.dumps({
        "model": model_name,
        "adapter": adapter,
        "benchmark": "MMBench Dev EN",
        "evaluation_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "average_generation_latency_sec": avg_latency,
        "peak_inference_vram_gb": peak_vram,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        "note": "Fixed project subset; not official MMBench leaderboard score.",
    }, indent=2), encoding="utf-8")

    print(f"Correct: {correct}/{len(results)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Avg latency: {avg_latency:.4f} s")
    print(f"Peak inference VRAM: {peak_vram:.2f} GB")


def extract_question(record):
    """Use the first human/user conversation turn as the student prompt."""
    for turn in record.get("conversations", []):
        if turn.get("from") in ("human", "user"):
            return turn.get("value", "").replace("<image>", "").strip()
    return ""


def make_inputs(processor, image, question, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    x = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in x.items()}


def student_labels(processor, image, question, teacher_response, device):
    """Build labels so CE is computed on teacher response, not the prompt."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + teacher_response

    prompt = processor(
        text=[prompt_text], images=[image], padding=True, return_tensors="pt"
    )
    full = processor(
        text=[full_text], images=[image], padding=True, return_tensors="pt"
    )
    full = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full.items()}
    labels = full["input_ids"].clone()
    prompt_len = prompt["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    if "attention_mask" in full:
        labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return full


def train_kd():
    """Run the main 1,000-step online response-based KD experiment."""
    ensure_dirs()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    records = json.loads(TRAIN_SUBSET.read_text(encoding="utf-8"))
    print("Training records:", len(records))

    teacher_processor = AutoProcessor.from_pretrained(TEACHER)
    teacher = Qwen3VLForConditionalGeneration.from_pretrained(
        TEACHER, torch_dtype=DTYPE, device_map="cuda"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_processor = AutoProcessor.from_pretrained(STUDENT)
    student = Qwen3VLForConditionalGeneration.from_pretrained(
        STUDENT, torch_dtype=DTYPE, device_map="cuda"
    )
    student.config.use_cache = False
    student.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_cfg)
    student.print_trainable_parameters()

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    device = next(student.parameters()).device
    losses = []
    start = time.perf_counter()
    KD_DIR.mkdir(parents=True, exist_ok=True)

    response_file = KD_DIR / "online_kd_teacher_responses.jsonl"
    with response_file.open("w", encoding="utf-8") as rf:
        for step in range(STEPS):
            record = records[step % len(records)]
            image_name = record["image"]
            image = Image.open(IMAGE_DIR / image_name).convert("RGB")
            question = extract_question(record)

            # ONLINE teacher generation
            teacher_inputs = make_inputs(
                teacher_processor, image, question, device
            )
            with torch.no_grad():
                generated = teacher.generate(
                    **teacher_inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            input_len = teacher_inputs["input_ids"].shape[1]
            teacher_new = generated[:, input_len:]
            teacher_response = teacher_processor.batch_decode(
                teacher_new, skip_special_tokens=True
            )[0].strip()

            rf.write(json.dumps({
                "step": step + 1,
                "image": image_name,
                "question": question,
                "teacher_response": teacher_response,
            }, ensure_ascii=False) + "\n")
            rf.flush()

            # Student response CE
            student_inputs = student_labels(
                student_processor, image, question, teacher_response, device
            )
            student.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = student(**student_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            value = float(loss.detach().cpu())
            losses.append(value)
            if (step + 1) == 1 or (step + 1) % 10 == 0:
                print(f"step {step+1}/{STEPS} loss={value:.6f}")

            if (step + 1) % 100 == 0:
                d = CKPT_DIR / f"step_{step+1}"
                d.mkdir(parents=True, exist_ok=True)
                student.save_pretrained(d)
                student_processor.save_pretrained(d)
                torch.save({"step": step + 1, "optimizer": optimizer.state_dict()}, d / "training_state.pt")

    final = CKPT_DIR / "final"
    final.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final)
    student_processor.save_pretrained(final)

    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())

    summary = {
        "teacher": TEACHER,
        "student": STUDENT,
        "method": "online_response_based_kd",
        "precision": "float16",
        "training_samples": len(records),
        "optimization_steps": STEPS,
        "average_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "training_time_sec": elapsed,
        "training_time_min": elapsed / 60,
        "peak_vram_gb": peak,
        "trainable_parameters": trainable,
        "total_student_parameters": total,
        "checkpoint": str(final),
        "teacher_responses": str(response_file),
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
        },
    }
    (KD_DIR / "online_kd_training_summary.json").write_text(json.dumps(summary, indent=2))
    (KD_DIR / "online_kd_config.json").write_text(json.dumps({
        "teacher": TEACHER, "student": STUDENT,
        "method": "online_response_based_kd", "precision": "float16",
        "steps": STEPS, "learning_rate": LR,
        "teacher_max_new_tokens": 64,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": TARGET_MODULES},
    }, indent=2))
    print(json.dumps(summary, indent=2))


def summary():
    """Print the final comparison from saved evaluation summaries."""
    paths = {
        "8B Teacher": RESULTS_DIR / "baseline_8b/mmbench_1000_summary.json",
        "2B Baseline": RESULTS_DIR / "baseline_2b/mmbench_1000_summary.json",
        "2B + Online KD": RESULTS_DIR / "kd/mmbench_1000_summary.json",
    }
    data = {k: json.loads(v.read_text()) for k, v in paths.items()}
    print("\nModel                    Accuracy   Latency(s)   VRAM(GB)")
    print("-" * 60)
    for k, x in data.items():
        print(f"{k:<24} {x['accuracy']*100:>7.2f}% {x['average_generation_latency_sec']:>11.4f} {x['peak_inference_vram_gb']:>10.2f}")

    t = data["8B Teacher"]
    b = data["2B Baseline"]
    k = data["2B + Online KD"]
    print("\nKD accuracy improvement:", f"{(k['accuracy']-b['accuracy'])*100:.2f} pp")
    print("Remaining teacher gap:", f"{(t['accuracy']-k['accuracy'])*100:.2f} pp")
    print("Original gap recovered:", f"{(k['accuracy']-b['accuracy'])/(t['accuracy']-b['accuracy'])*100:.1f}%")
    print("KD / teacher accuracy:", f"{k['accuracy']/t['accuracy']*100:.1f}%")
    print("KD VRAM reduction vs teacher:", f"{(1-k['peak_inference_vram_gb']/t['peak_inference_vram_gb'])*100:.1f}%")
    print("KD latency reduction vs teacher:", f"{(1-k['average_generation_latency_sec']/t['average_generation_latency_sec'])*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["prepare_train", "download_images", "validate_train", "prepare_eval", "train_kd", "summary"]:
        sub.add_parser(name)

    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--adapter", default=None)
    e.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "prepare_train": prepare_train()
    elif args.cmd == "download_images": download_images()
    elif args.cmd == "validate_train": validate_train()
    elif args.cmd == "prepare_eval": prepare_eval()
    elif args.cmd == "train_kd": train_kd()
    elif args.cmd == "summary": summary()
    elif args.cmd == "eval": evaluate(args.model, args.adapter, args.out)


if __name__ == "__main__":
    main()
