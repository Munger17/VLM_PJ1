"""
train.py — MVP Fine-Tuning Script
Loads Qwen3-VL-4B-Instruct in 4-bit, attaches LoRA adapters,
and trains with HuggingFace Trainer on fashionpedia_4_categories.

Key design: raw dataset (with top-level PIL images) is loaded,
and formatting into Qwen3-VL chat template happens inside the
collate_fn at training time. This avoids PIL serialization issues
with nested message structures.

Usage:
    python train.py                           # default settings
    python train.py --epochs 3 --lr 2e-4      # override
"""

import argparse
import os
import torch
import psutil
from datasets import load_from_disk
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model


class HardwareMonitorCallback(TrainerCallback):
    """Logs GPU & CPU usage to TensorBoard at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics = {}

        # GPU metrics (CUDA)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # VRAM usage
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_mem / 1024**3
                metrics[f"hw/gpu{i}_vram_allocated_gb"] = round(allocated, 2)
                metrics[f"hw/gpu{i}_vram_reserved_gb"] = round(reserved, 2)
                metrics[f"hw/gpu{i}_vram_total_gb"] = round(total, 2)
                metrics[f"hw/gpu{i}_vram_percent"] = round(allocated / total * 100, 1)

                # GPU utilization (requires pynvml)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f"hw/gpu{i}_utilization_percent"] = util.gpu
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f"hw/gpu{i}_temp_celsius"] = temp
                except Exception:
                    pass  # pynvml not installed, skip GPU util

        # CPU & RAM metrics
        metrics["hw/cpu_percent"] = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        metrics["hw/ram_used_gb"] = round(ram.used / 1024**3, 2)
        metrics["hw/ram_percent"] = ram.percent

        # Write to TensorBoard via the trainer's log method
        if logs is not None:
            logs.update(metrics)


# Category ID → human-readable name
CATEGORY_NAMES = {0: "clothing", 1: "bags", 2: "accessories", 3: "shoes"}

SYSTEM_MESSAGE = (
    "You are a fashion object detection assistant. "
    "Given an image of a person or outfit, identify all visible fashion items "
    "and provide their bounding boxes as [x_min, y_min, x_max, y_max] in pixels."
)

USER_PROMPT = (
    "Detect all fashion items in this image. "
    "List each item with its category and bounding box."
)


def format_bbox_answer(objects: dict) -> str:
    """Convert raw bbox annotations into a structured text answer."""
    lines = []
    for cat_id, bbox in zip(objects["category"], objects["bbox"]):
        name = CATEGORY_NAMES.get(cat_id, "unknown")
        box_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        lines.append(f"- {name}: {box_str}")
    return "\n".join(lines) if lines else "No fashion items detected."


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-4B-Instruct")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./output/qwen3vl-4b-fashion-lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    # ── 1. Load model in 4-bit (QLoRA) ──────────────────────────────
    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_id} in 4-bit...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # ── 2. Attach LoRA adapters ─────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── 3. Load raw data ────────────────────────────────────────────
    print(f"Loading data from {args.data_dir}...")
    train_dataset = load_from_disk(f"{args.data_dir}/train")
    val_dataset = load_from_disk(f"{args.data_dir}/val")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # ── 4. Training config ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "tb_logs"),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # ── 5. Collator: raw dataset → chat template → tokenized batch ──
    def collate_fn(examples):
        texts = []
        images = []

        for ex in examples:
            # Get the PIL image (top-level column, deserializes fine)
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

            # Build ground truth answer from annotations
            answer = format_bbox_answer(ex["objects"])

            # Build chat messages for this sample
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # placeholder — actual image passed separately
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]

            # Apply chat template → text string
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        # Tokenize text + process images together
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        # Labels = input_ids; mask padding tokens with -100
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    # ── 6. Train ────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[HardwareMonitorCallback()],
    )

    print("Starting training...")
    trainer.train()

    # ── 7. Save LoRA adapter ────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()