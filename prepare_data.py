"""
prepare_data.py — MVP Data Preparation
Downloads fashionpedia_4_categories from HuggingFace,
slices to N samples, and saves the RAW dataset to disk.
Formatting into chat template happens at training time
(PIL images don't serialize well inside nested message dicts).

Usage:
    python prepare_data.py                    # default 500 train / 100 val
    python prepare_data.py --n_train 200 --n_val 50
"""

import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare fashionpedia data for fine-tuning")
    parser.add_argument("--n_train", type=int, default=500, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--save_dir", type=str, default="./data/raw", help="Output directory")
    args = parser.parse_args()

    print("Loading fashionpedia_4_categories from HuggingFace...")
    ds = load_dataset("detection-datasets/fashionpedia_4_categories")

    # Slice
    train_ds = ds["train"].select(range(min(args.n_train, len(ds["train"]))))
    val_ds = ds["val"].select(range(min(args.n_val, len(ds["val"]))))

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Save raw dataset (images serialize fine at top level)
    train_ds.save_to_disk(f"{args.save_dir}/train")
    val_ds.save_to_disk(f"{args.save_dir}/val")

    print(f"Saved to {args.save_dir}/")

    # Quick sanity check
    sample = train_ds[0]
    print(f"\nSanity check — sample image_id: {sample['image_id']}")
    print(f"  Image size: {sample['image'].size}, mode: {sample['image'].mode}")
    print(f"  Num objects: {len(sample['objects']['category'])}")
    print(f"  Categories: {sample['objects']['category']}")
    print(f"  Bboxes: {sample['objects']['bbox'][:2]}...")


if __name__ == "__main__":
    main()