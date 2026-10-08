"""Offline hard-response KD using the teacher-response cache from online KD."""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


TEACHER_NAME = "Qwen/Qwen3-VL-8B-Instruct"
STUDENT_NAME = "Qwen/Qwen3-VL-2B-Instruct"
TRAIN_PATH = os.environ.get("KD_TRAIN_PATH", "data/train_llava/train_10000.json")
IMAGE_DIR = Path("data/train_llava/images")
TEACHER_RESPONSES_PATH = Path(
    "results/kd_online_10k/online_kd_teacher_responses.jsonl"
)
CHECKPOINT_DIR = Path(os.environ.get("KD_CHECKPOINT_DIR", "checkpoints/offline_kd_2b_10k"))
RESULT_DIR = Path(os.environ.get("KD_RESULT_DIR", "results/kd_offline_10k"))
LOG_DIR = Path(os.environ.get("KD_LOG_DIR", "logs/offline_10k"))
MAX_STEPS = int(os.environ.get("KD_MAX_STEPS", "10000"))
SAVE_EVERY = int(os.environ.get("KD_SAVE_EVERY", "500"))
LEARNING_RATE = 2e-5
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
DEVICE = "cuda:0"
TRAINING_SEED = 123

random.seed(TRAINING_SEED)
torch.manual_seed(TRAINING_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TRAINING_SEED)


def first_human_turn(record):
    for turn in record["conversations"]:
        if turn["from"] == "human":
            return turn["value"].replace("<image>", "").strip()
    raise ValueError(f"No human turn in record {record['id']}")


def build_user_message(image, question):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


print("=" * 80)
print("OFFLINE KNOWLEDGE DISTILLATION TRAINING")
print("=" * 80)
print("Start:", datetime.now().isoformat())
print("Teacher:", TEACHER_NAME)
print("Student:", STUDENT_NAME)
print("KD type: offline cached hard-response KD")
print("Max steps:", MAX_STEPS)

with open(TRAIN_PATH, "r", encoding="utf-8") as handle:
    records = json.load(handle)
with TEACHER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
    cached_responses = [json.loads(line) for line in handle if line.strip()]

if len(records) != 10000:
    raise ValueError(f"Expected exactly 10000 training records, found {len(records)}")
if len(cached_responses) != len(records):
    raise ValueError(f"Expected {len(records)} cached responses, found {len(cached_responses)}")
if not 1 <= MAX_STEPS <= len(records):
    raise ValueError(f"KD_MAX_STEPS must be between 1 and {len(records)}")
if SAVE_EVERY < 1:
    raise ValueError("KD_SAVE_EVERY must be at least 1")
for index, (record, cached) in enumerate(zip(records, cached_responses)):
    if str(record["id"]) != str(cached["record_id"]):
        raise ValueError(f"Cache order mismatch at position {index}")
    if not cached["teacher_response"].strip():
        raise ValueError(f"Empty cached teacher response at position {index}")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

config = {
    "teacher": TEACHER_NAME, "student": STUDENT_NAME,
    "method": "offline_hard_response_kd", "teacher_used_during_training": False,
    "student_adaptation": "LoRA", "precision": "float16", "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0), "train_records": len(records),
    "training_seed": TRAINING_SEED, "checkpoint_every_steps": SAVE_EVERY,
    "learning_rate": LEARNING_RATE, "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
    "optimizer": "AdamW", "teacher_response_cache": str(TEACHER_RESPONSES_PATH),
    "max_steps": MAX_STEPS, "evaluation_set": "data/eval_1000/MMBench_DEV_EN_1000.tsv",
    "created": datetime.now().isoformat(),
}
save_json(RESULT_DIR / "offline_kd_config.json", config)

student = Qwen3VLForConditionalGeneration.from_pretrained(
    STUDENT_NAME, torch_dtype=torch.float16, device_map=DEVICE
)
student_processor = AutoProcessor.from_pretrained(STUDENT_NAME)
student = get_peft_model(student, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
))
student.config.use_cache = False
student.train()
student.print_trainable_parameters()
trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in student.parameters())
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=LEARNING_RATE
)
torch.cuda.reset_peak_memory_stats()
loss_history = []
teacher_responses = []
training_start = time.perf_counter()

for step in range(MAX_STEPS):
    record = records[step]
    question = first_human_turn(record)
    image = Image.open(IMAGE_DIR / record["image"]).convert("RGB")
    messages = build_user_message(image, question)
    cached_item = cached_responses[step]
    teacher_response = cached_item["teacher_response"].strip()
    student_messages = messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": teacher_response}],
    }]
    prompt_text = student_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = student_processor.apply_chat_template(
        student_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_inputs = student_processor(
        text=[prompt_text], images=[image], return_tensors="pt"
    ).to(DEVICE)
    full_inputs = student_processor(
        text=[full_text], images=[image], return_tensors="pt"
    ).to(DEVICE)
    labels = full_inputs.input_ids.clone()
    labels[:, :prompt_inputs.input_ids.shape[1]] = -100
    if student_processor.tokenizer.pad_token_id is not None:
        labels[labels == student_processor.tokenizer.pad_token_id] = -100
    full_inputs["labels"] = labels
    optimizer.zero_grad(set_to_none=True)
    outputs = student(**full_inputs)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    loss_history.append(loss_value)
    teacher_responses.append({
        "step": step + 1, "record_id": record["id"], "image": record["image"],
        "question": question, "teacher_response": teacher_response, "loss": loss_value,
    })
    print(f"[{step + 1:4d}/{MAX_STEPS}] loss={loss_value:.6f}", flush=True)
    if (step + 1) % SAVE_EVERY == 0:
        checkpoint = CHECKPOINT_DIR / f"step_{step + 1}"
        student.save_pretrained(checkpoint)
        student_processor.save_pretrained(checkpoint)
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        save_json(checkpoint / "training_state.json", {
            "step": step + 1, "loss": loss_value, "timestamp": datetime.now().isoformat()
        })
    image.close()
    del prompt_inputs, full_inputs, outputs

training_time = time.perf_counter() - training_start
peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
final_dir = CHECKPOINT_DIR / "final"
student.save_pretrained(final_dir)
student_processor.save_pretrained(final_dir)
with open(RESULT_DIR / "offline_kd_used_teacher_responses.jsonl", "w", encoding="utf-8") as handle:
    for item in teacher_responses:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
summary = {
    "teacher": TEACHER_NAME, "student": STUDENT_NAME,
    "method": "offline_hard_response_kd", "precision": "float16",
    "training_samples": MAX_STEPS, "average_loss": sum(loss_history) / len(loss_history),
    "first_loss": loss_history[0], "final_loss": loss_history[-1],
    "training_time_sec": training_time, "training_time_min": training_time / 60,
    "peak_vram_gb": peak_vram, "trainable_parameters": trainable_params,
    "total_student_parameters": total_params, "checkpoint": str(final_dir),
    "teacher_responses": str(RESULT_DIR / "offline_kd_used_teacher_responses.jsonl"),
    "end_time": datetime.now().isoformat(),
}
save_json(RESULT_DIR / "offline_kd_training_summary.json", summary)
print("OFFLINE KD TRAINING FINISHED")
print("Average loss:", summary["average_loss"])
print("Final checkpoint:", final_dir)
