"""Expectation-equivalent 75% ground-truth / 25% cached-teacher KD."""

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
TEACHER_RESPONSES_PATH = Path("results/kd_online_10k/online_kd_teacher_responses.jsonl")
CHECKPOINT_DIR = Path(os.environ.get("KD_CHECKPOINT_DIR", "checkpoints/offline_mixed_kd_2b_10k_a075"))
RESULT_DIR = Path(os.environ.get("KD_RESULT_DIR", "results/kd_offline_mixed_10k_a075"))
LOG_DIR = Path(os.environ.get("KD_LOG_DIR", "logs/offline_mixed_10k_a075"))
MAX_STEPS = int(os.environ.get("KD_MAX_STEPS", "10000"))
SAVE_EVERY = int(os.environ.get("KD_SAVE_EVERY", "500"))
LEARNING_RATE = float(os.environ.get("KD_LEARNING_RATE", "1e-5"))
GROUND_TRUTH_WEIGHT = float(os.environ.get("KD_GROUND_TRUTH_WEIGHT", "0.75"))
TEACHER_WEIGHT = 1.0 - GROUND_TRUTH_WEIGHT
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
DEVICE = "cuda:0"
TRAINING_SEED = 123

random.seed(TRAINING_SEED)
torch.manual_seed(TRAINING_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TRAINING_SEED)


def first_training_pair(record):
    conversations = record["conversations"]
    for index, turn in enumerate(conversations):
        if turn["from"] != "human":
            continue
        if index + 1 >= len(conversations) or conversations[index + 1]["from"] != "gpt":
            raise ValueError(f"Human turn lacks following GPT answer in record {record['id']}")
        question = turn["value"].replace("<image>", "").strip()
        answer = conversations[index + 1]["value"].strip()
        if not question or not answer:
            raise ValueError(f"Empty question/answer in record {record['id']}")
        return question, answer
    raise ValueError(f"No human/GPT pair in record {record['id']}")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


with open(TRAIN_PATH, "r", encoding="utf-8") as handle:
    records = json.load(handle)
with TEACHER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
    cached_responses = [json.loads(line) for line in handle if line.strip()]
if len(records) != 10000 or len(cached_responses) != len(records):
    raise ValueError("Training data/cache must contain the same 10,000 ordered records")
if not 1 <= MAX_STEPS <= len(records) or SAVE_EVERY < 1:
    raise ValueError("Invalid KD_MAX_STEPS or KD_SAVE_EVERY")
if not 0.0 <= GROUND_TRUTH_WEIGHT <= 1.0:
    raise ValueError("KD_GROUND_TRUTH_WEIGHT must be in [0, 1]")
for index, (record, cached) in enumerate(zip(records, cached_responses)):
    first_training_pair(record)
    if str(record["id"]) != str(cached["record_id"]):
        raise ValueError(f"Cache order mismatch at position {index}")
    if not cached["teacher_response"].strip():
        raise ValueError(f"Empty teacher response at position {index}")

ground_truth_steps = round(MAX_STEPS * GROUND_TRUTH_WEIGHT)
teacher_steps = MAX_STEPS - ground_truth_steps
target_schedule = ["ground_truth"] * ground_truth_steps + ["teacher"] * teacher_steps
random.Random(TRAINING_SEED).shuffle(target_schedule)
for kind, directory in (("checkpoint", CHECKPOINT_DIR), ("result", RESULT_DIR), ("log", LOG_DIR)):
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing {kind} directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)

config = {
    "teacher": TEACHER_NAME, "student": STUDENT_NAME,
    "method": "stochastic_mixed_supervised_offline_hard_response_kd",
    "mixture_strategy": "fixed-seed shuffled exact-quota mixture; one target per step",
    "ground_truth_weight": GROUND_TRUTH_WEIGHT, "teacher_weight": TEACHER_WEIGHT,
    "scheduled_ground_truth_steps": ground_truth_steps, "scheduled_teacher_steps": teacher_steps,
    "teacher_used_during_training": False, "student_adaptation": "LoRA",
    "precision": "float16", "device": DEVICE, "gpu": torch.cuda.get_device_name(0),
    "train_records": len(records), "training_seed": TRAINING_SEED,
    "checkpoint_every_steps": SAVE_EVERY, "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW", "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT, "teacher_response_cache": str(TEACHER_RESPONSES_PATH),
    "max_steps": MAX_STEPS, "created": datetime.now().isoformat(),
}
save_json(RESULT_DIR / "offline_mixed_kd_config.json", config)
print("Ground-truth steps:", ground_truth_steps, "Teacher steps:", teacher_steps)

student = Qwen3VLForConditionalGeneration.from_pretrained(
    STUDENT_NAME, torch_dtype=torch.float16, device_map=DEVICE
)
processor = AutoProcessor.from_pretrained(STUDENT_NAME)
student = get_peft_model(student, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM"
))
student.config.use_cache = False
student.train()
trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in student.parameters())
optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LEARNING_RATE)
torch.cuda.reset_peak_memory_stats()
loss_history = []
losses_by_target = {"ground_truth": [], "teacher": []}
training_records = []
training_start = time.perf_counter()

for step in range(MAX_STEPS):
    record = records[step]
    question, ground_truth = first_training_pair(record)
    teacher_response = cached_responses[step]["teacher_response"].strip()
    target_type = target_schedule[step]
    target_response = ground_truth if target_type == "ground_truth" else teacher_response
    image = Image.open(IMAGE_DIR / record["image"]).convert("RGB")
    user_messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}
    ]}]
    full_messages = user_messages + [{"role": "assistant", "content": [
        {"type": "text", "text": target_response}
    ]}]
    prompt_text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(DEVICE)
    full_inputs = processor(text=[full_text], images=[image], return_tensors="pt").to(DEVICE)
    prompt_length = prompt_inputs.input_ids.shape[1]
    full_length = full_inputs.input_ids.shape[1]
    if full_length <= prompt_length:
        raise RuntimeError(f"Target token boundary failed at step {step + 1}")
    labels = full_inputs.input_ids.clone()
    labels[:, :prompt_length] = -100
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    supervised_mask = labels.ne(-100)
    supervised_tokens = int(supervised_mask.sum().item())
    if supervised_tokens == 0:
        raise RuntimeError(f"No supervised tokens at step {step + 1}")
    if step == 0:
        decoded_target = processor.tokenizer.decode(labels[supervised_mask], skip_special_tokens=False)
        print("BOUNDARY CHECK PASSED")
        print("Prompt tokens:", prompt_length, "Full tokens:", full_length)
        print("Supervised tokens:", supervised_tokens)
        print("Decoded supervised suffix:", repr(decoded_target[:500]))
    full_inputs["labels"] = labels
    optimizer.zero_grad(set_to_none=True)
    outputs = student(**full_inputs)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    loss_history.append(loss_value)
    losses_by_target[target_type].append(loss_value)
    training_records.append({
        "step": step + 1, "record_id": record["id"], "image": record["image"],
        "question": question, "ground_truth": ground_truth,
        "teacher_response": teacher_response, "target_type": target_type,
        "target_response": target_response, "supervised_tokens": supervised_tokens,
        "loss": loss_value,
    })
    print(f"[{step + 1:4d}/{MAX_STEPS}] target={target_type} loss={loss_value:.6f}", flush=True)
    if (step + 1) % SAVE_EVERY == 0:
        checkpoint = CHECKPOINT_DIR / f"step_{step + 1}"
        student.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        save_json(checkpoint / "training_state.json", {
            "step": step + 1, "loss": loss_value, "target_type": target_type,
            "timestamp": datetime.now().isoformat(),
        })
    image.close()
    del prompt_inputs, full_inputs, outputs

training_time = time.perf_counter() - training_start
peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
final_dir = CHECKPOINT_DIR / "final"
student.save_pretrained(final_dir)
processor.save_pretrained(final_dir)
with open(RESULT_DIR / "mixed_kd_training_records.jsonl", "w", encoding="utf-8") as handle:
    for item in training_records:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
summary = {
    "teacher": TEACHER_NAME, "student": STUDENT_NAME,
    "method": "stochastic_mixed_supervised_offline_hard_response_kd",
    "ground_truth_weight": GROUND_TRUTH_WEIGHT, "teacher_weight": TEACHER_WEIGHT,
    "ground_truth_steps": len(losses_by_target["ground_truth"]),
    "teacher_steps": len(losses_by_target["teacher"]),
    "average_ground_truth_loss": sum(losses_by_target["ground_truth"]) / max(1, len(losses_by_target["ground_truth"])),
    "average_teacher_loss": sum(losses_by_target["teacher"]) / max(1, len(losses_by_target["teacher"])),
    "training_samples": MAX_STEPS, "average_loss": sum(loss_history) / len(loss_history),
    "first_loss": loss_history[0], "final_loss": loss_history[-1],
    "training_time_sec": training_time, "training_time_min": training_time / 60,
    "peak_vram_gb": peak_vram, "trainable_parameters": trainable_params,
    "total_student_parameters": total_params, "checkpoint": str(final_dir),
    "training_records": str(RESULT_DIR / "mixed_kd_training_records.jsonl"),
    "end_time": datetime.now().isoformat(),
}
save_json(RESULT_DIR / "offline_kd_training_summary.json", summary)
print("OFFLINE MIXED KD TRAINING FINISHED")
print("Average loss:", summary["average_loss"])
print("Final checkpoint:", final_dir)
