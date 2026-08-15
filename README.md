# MATS 12.0 — Does “Wait” Mean the Model Corrected Itself?

This repository contains the reproducibility package for a scoped MATS 12.0 application project on chain-of-thought faithfulness and backtracking in reasoning models.

The empirical pilot analyzes 2,000 real reasoning traces from the public [OpenThoughts-114k dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k). It measures the prevalence of `Wait`, compares local correction-language proxies with matched random-position windows, saves randomly sampled raw traces, and renders three graphs. The result is deliberately observational: it does **not** claim that `Wait` is causally responsible for a correction or that the written CoT is faithful.

## Reproduction

Install the small analysis dependencies and run:

```bash
pip install requests pandas matplotlib
python3 analyze_openthoughts.py
```

The script uses the public Hugging Face datasets-server API. It samples twenty evenly spaced 100-row slices from the metadata train split, uses seed `20260815`, and writes derived artifacts under `openthoughts_analysis/`.

## Main artifacts

| File | Description |
|---|---|
| `mats12_research_writeup.md` | Full write-up with the executive summary first |
| `mats12_experiment_protocol.md` | Scoped protocol and proposed target-model causal extension |
| `analyze_openthoughts.py` | Reproducible data-download, analysis, and plotting script |
| `openthoughts_analysis/summary.json` | Headline statistics and paths |
| `openthoughts_analysis/qualitative_samples.jsonl` | Fixed-seed raw examples for manual audit |
| `openthoughts_analysis/*.png` | Experimental figures |

The target-model extension is designed for DeepSeek-R1-Distill-Qwen-7B and Qwen2.5-7B-Instruct, but those models were not run in the CPU-only sandbox. Any target-specific causal result must therefore be generated in a GPU environment or from supplied rollouts; this repository does not silently substitute another model.

## Primary sources

- [Chua, Evans et al., Are DeepSeek R1 and other reasoning models more faithful?](https://arxiv.org/html/2501.08156v4)
- [DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/html/2501.12948)
- [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
