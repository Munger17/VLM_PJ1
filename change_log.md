# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-08

### Added
- MVP fine-tuning pipeline: `prepare_data.py`, `train.py`, `inference.py`
- QLoRA (4-bit NF4) training on Qwen3-VL-4B-Instruct
- fashionpedia_4_categories dataset (4-class fashion object detection)
- TensorBoard logging with hardware monitoring (GPU VRAM, utilization, CPU, RAM)
- `export_tb_logs.py` for offline log export to CSV + PNG plots
- GitHub CI (ruff lint), issue templates, PR template