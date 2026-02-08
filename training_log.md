# Training Log

## Experiment 1 — MVP Baseline

**Date**: 2026-02-08
**Commit**: (paste commit hash)

### Config
| Param | Value |
|---|---|
| Model | Qwen3-VL-4B-Instruct |
| Method | QLoRA (NF4, rank 16) |
| Dataset | fashionpedia_4_categories |
| Train samples | 500 |
| Val samples | 100 |
| Epochs | 1 |
| Batch size | 1 × 8 grad accum |
| LR | 1e-4, cosine schedule |
| Max length | 2048 |

### Hardware
| Resource | Value |
|---|---|
| GPU | (your GPU here) |
| VRAM peak | (fill after training) |
| Training time | (fill after training) |

### Results
| Metric | Value |
|---|---|
| Final train loss | (fill) |
| Final eval loss | (fill) |

### Observations
<!-- What worked, what didn't, what to try next -->

---

## Experiment 2 — (next experiment title)

(copy the template above)