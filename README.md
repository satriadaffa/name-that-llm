# Name That LLM — Authorship Attribution of 18 Frontier Models

My solution work for the Kaggle competition [**Name That LLM**](https://www.kaggle.com/competitions/name-that-llm)
(Google / Kaggle, Oct 2026). The task: given a short, unlabelled paragraph (40–250 words), predict which of
**18 frontier LLMs** (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot) wrote it. Scored on **balanced accuracy**;
the test prompts never appear in the training set.

This repo is a record of my experiments and reasoning, not a one-shot winning script. It is deliberately honest
about what worked, what didn't, and which parts build on public community work.

## TL;DR results (local CV = GroupKFold by prompt, scored on competition passages only)

| Step | Model | CV balanced acc | Note |
|---|---|---|---|
| Random guess | — | 0.056 | 1/18 |
| Baseline | TF-IDF (char+word) + LogReg | **0.342** | LB 0.345 → **CV matched LB within 0.003** |
| + prompt-group Hungarian | one model per prompt cluster | 0.436 (CV) | LB flat → honest overfit to simulated test layout |
| Multi-view n-gram | char / word / word-shape / function-word views + LinearSVC | **0.385** | host data only |
| + public extra data | prvsiyan Extended dataset | 0.392 | CC BY-SA 4.0 |
| + larger public pool | ~35k deduped public passages | **0.441** | data is the strongest single lever |
| Pseudo-labelling | self-training on confident test preds | 0.422 ❌ | **negative result** — kept for the record |

Final leaderboard submission (~0.60) is a **fork** of a public two-stage LoRA notebook — see *Attribution* below.
My own contribution is the analysis, the multi-view model, the data curation/generation pipeline, and the
validation discipline that let me trust local CV over the noisy public leaderboard.

## What I'd want a reviewer to notice

- **Validation that tracks the leaderboard.** The test prompts are unseen, so random KFold leaks topic and
  flatters the score (0.22 random-split vs 0.37 prompt-split in my tests). I used **GroupKFold on `prompt_id`**
  throughout; the baseline's CV (0.342) landed within 0.003 of the public LB (0.345). After that I trusted CV
  and stopped burning submissions.
- **A negative result reported honestly.** The "one model per prompt" Hungarian assignment looked great in
  simulation (+0.09 CV) but gave nothing on the real LB, because the hidden test's per-model counts aren't
  uniform. I kept it visible rather than hiding it. Same for pseudo-labelling, which *lowered* CV.
- **Data is the lever, measured not assumed.** Swapping a small hand-made extra set for a large deduped public
  pool moved the n-gram model from 0.392 to 0.441 — a bigger jump than any modelling change.

## The models

**Multi-view n-gram + LinearSVC** (`src/cv_multiview.py`). Four TF-IDF views, each capturing a different
fingerprint and fused with per-view weights:
- **char 3–5** — punctuation and sub-word habits
- **word 1–3** — phrasing and collocations
- **word-shape 2–6** — capitalisation/length rhythm with content removed (`Xxxx xx X.`)
- **function-word 1–3** — syntax with content words masked to `CONTENT`, so topic can't leak

The hardest confusions are *within* a model family (e.g. `gemini-3.6/3.7/3.8-flash`, `gpt-5.6-luna/sol/terra`)
— see `src/assign_test.py` for the per-class error analysis.

## Synthetic data pipeline

`src/pilot_generate.py`, `src/kbench_cells.py`, `src/pilot_validate.py`. The competition ships only 150 prompts,
so extra data is valuable. I built a generator over Kaggle Benchmarks' free model proxy (12 of the 18 models
reachable, ~\$0.015/passage), matching the host's format (single paragraph, no markdown, per-passage length
target). `pilot_validate.py` checks that a host-trained classifier scores generated passages about the same as
real ones (0.394 vs 0.393) — i.e. the synthetic data is in-distribution before it's ever used for training.

## Repo map

```
src/
  baseline.py          TF-IDF + LogReg/LinearSVC baseline, GroupKFold CV
  assign_test.py       prompt-cluster Hungarian assignment + per-class error analysis
  cv_multiview.py      multi-view n-gram model (the main reusable module)
  cv_svc_sweep.py      C / view-weight sweep
  cv_pool_compare.py   which public data pool helps most
  cv_with_extra.py     effect of adding generated data, with sample weighting
  cv_pseudo.py         pseudo-labelling experiment (negative result)
  pool_source_check.py per-source distribution check of the public pool
  pilot_generate.py    synthetic passage generation via Kaggle Benchmarks proxy
  kbench_cells.py      the cells run inside the Kaggle Benchmarks task notebook
  pilot_validate.py    format + label-match validation of generated data
notebooks/
  submit_baseline.ipynb    baseline submission (LB 0.345)
  submit_multiview.ipynb   multi-view + public data submission
```

Competition data and generated/pool data are **not** committed (see `.gitignore`) — the competition rules
forbid redistributing competition data, and the public datasets have their own pages.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# place train.csv / test.csv under data/, public pools under data/external/
.venv/Scripts/python src/cv_multiview.py        # host / +public / +public+own CV
```

## Attribution

- Final ~0.60 LB submission is a fork of a **public, Apache-2.0** two-stage LoRA notebook by the Kaggle
  community (Qwen3-8B/14B LoRA + Gaussian shared-prompt core + Hungarian). Full credit to the original
  authors; I do not claim that pipeline as my own.
- Public extra-data pools used: *prvsiyan's Extended Name That LLM Dataset* and *NTL two-stage LoRA public
  pool* (both CC BY-SA 4.0), built from many community contributors' generations.
- My original work: the experiments, analysis, multi-view model, synthetic-data pipeline, and this writeup.
