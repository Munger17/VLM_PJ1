"""
inference.py — MVP Smoke Test
Loads the base Qwen3-VL-4B-Instruct + trained LoRA adapter,
runs inference on a single image to verify the pipeline works.

Usage:
    python inference.py                              # uses a sample from val set
    python inference.py --image_path test.jpg        # use your own image
"""

import argparse
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./output/qwen3vl-4b-fashion-lora",
        help="Path to LoRA adapter",
    )
    parser.add_argument("--image_path", type=str, default=None, help="Path to test image")
    parser.add_argument("--base_only", action="store_true", help="Run base model without adapter (for comparison)")
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading base model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if not args.base_only:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    processor = AutoProcessor.from_pretrained(model_id)

    # ── Load image ──────────────────────────────────────────────────
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        print(f"Loaded image: {args.image_path}")
    else:
        # Pull one sample from the val set as a default test
        from datasets import load_dataset

        print("No image provided, loading a sample from fashionpedia val set...")
        ds = load_dataset("detection-datasets/fashionpedia_4_categories", split="val")
        sample = ds[0]
        image = sample["image"].convert("RGB")
        print(f"Using val sample image_id={sample['image_id']}")
        # Print ground truth for comparison
        from prepare_data import format_bbox_answer, CATEGORY_NAMES

        print(f"\n--- Ground Truth ---")
        print(format_bbox_answer(sample["objects"]))

    # ── Run inference ───────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Detect all fashion items in this image. "
                    "List each item with its category and bounding box.",
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print("\nGenerating...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"\n--- Model Output ---")
    print(output_text)


if __name__ == "__main__":
    main()