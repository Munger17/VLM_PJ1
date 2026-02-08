# VLM_PJ1 — Qwen3-VL-4B Fashion Object Detection Fine-Tuning

Fine-tune **Qwen3-VL-4B-Instruct** on [fashionpedia_4_categories](https://huggingface.co/datasets/detection-datasets/fashionpedia_4_categories) for fashion item detection (clothing, shoes, bags, accessories) with bounding boxes.

**MVP scope**: 4 files, end-to-end in 3 commands.

## Hardware

- **Minimum**: 1× GPU with 16GB VRAM (T4, RTX 3090) via QLoRA 4-bit
- **Recommended**: 1× GPU with 24GB VRAM (RTX 4090, A6000)

## Setup

```bash
git clone https://github.com/Munger17/VLM_PJ1.git
cd VLM_PJ1
pip install -r requirements.txt
```

## Run (3 commands)

```bash
# Step 1: Download & format 500 training images (~1 min)
python prepare_data.py

# Step 2: Fine-tune with QLoRA (~20-60 min depending on GPU)
python train.py

# Step 3: Smoke test inference
python inference.py
```

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `prepare_data.py` | Downloads dataset, formats to Qwen3-VL chat template |
| `train.py` | QLoRA fine-tuning via TRL SFTTrainer |
| `inference.py` | Load adapter + run on single image |

## Dataset

**fashionpedia_4_categories** — 42k images (we use 500 for MVP), 4 classes:
- Clothing (shirts, pants, dresses, jackets, etc.)
- Shoes
- Bags
- Accessories (glasses, hats, belts, scarves, etc.)

Each image has bounding box annotations `[x_min, y_min, x_max, y_max]` per object.

## Training Details

- **Model**: `Qwen/Qwen3-VL-4B-Instruct` (4.4B params, dense)
- **Method**: QLoRA (NF4 quantization + LoRA rank 16)
- **Trainable params**: ~0.5% of total
- **VRAM usage**: ~12-14GB

## What's Next (Post-MVP)

- [ ] Evaluation metrics (mAP, IoU)
- [ ] Config file (YAML) for hyperparameters
- [ ] Multi-GPU / DeepSpeed support
- [ ] W&B / TensorBoard logging
- [ ] CI/CD pipeline
- [ ] Gradio demo app