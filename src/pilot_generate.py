"""Pilot: generate host-style passages via the Kaggle Benchmarks model proxy.
Runs inside a Kaggle Benchmarks task notebook. Only kbench + pandas: installing scipy/sklearn there
crashes the kernel. Validation happens locally in src/pilot_validate.py."""
import os
import random
from concurrent.futures import ThreadPoolExecutor

import kaggle_benchmarks as kbench
import pandas as pd

# competition label -> substring of the proxy key
MODELS = {
    "gpt-5.5": "openai/gpt-5.5-", "gpt-5.6-luna": "openai/gpt-5.6-luna", "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra", "gpt-6-astra": "openai/gpt-6-astra",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview", "gemini-3.6-flash": "google/gemini-3.6-flash",
    "gemini-3.8-flash": "google/gemini-3.8-flash", "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "claude-opus-4-8": "anthropic/claude-opus-4-8", "claude-opus-5": "anthropic/claude-opus-5@",
    "claude-sonnet-5": "anthropic/claude-sonnet-5@",
}
KEYS = {label: next(k for k in kbench.llms if sub in k) for label, sub in MODELS.items()}

PROMPTS = [
    "Explain the Mpemba effect vividly for a general audience.",
    "Describe what it would feel like to fall into Jupiter's atmosphere.",
    "Explain Braess's paradox using an everyday analogy.",
    "Write a haunting short story about the last lighthouse keeper on a drowned coast.",
    "Explain how tardigrades survive the vacuum of space.",
    "Argue whether Theseus's ship is still the same ship after every plank is replaced.",
    "Explain the Monty Hall problem so it finally feels intuitive.",
    "Describe the Carrington Event of 1859 as if you were a telegraph operator.",
    "Explain how quorum sensing lets bacteria act as a crowd.",
    "Explain Goodhart's law with a vivid example.",
    "Write a whimsical description of a library where books rearrange themselves at night.",
    "Explain the Lindy effect and why old ideas tend to last.",
]
TEMPLATE = ("{prompt}\n\nWrite your answer as a single self-contained paragraph of about {n} words, "
            "with no headings, lists or markdown.")
OUT = "pilot_generated.csv"

random.seed(0)
jobs = [(p_id, p, label, random.randint(40, 250)) for p_id, p in enumerate(PROMPTS) for label in MODELS]
done = set()
if os.path.exists(OUT):  # resume: don't pay twice
    prev = pd.read_csv(OUT)
    done = set(zip(prev.prompt_id, prev.model))
jobs = [j for j in jobs if (f"pilot_{j[0]}", j[2]) not in done]
print(f"{len(jobs)} jobs to run", flush=True)


def run(job):
    p_id, prompt, label, n = job
    try:
        text = str(kbench.llms[KEYS[label]].prompt(TEMPLATE.format(prompt=prompt, n=n)))
    except Exception as e:  # keep going; report failures
        text = f"ERROR: {e}"
    return {"prompt_id": f"pilot_{p_id}", "model": label, "target_words": n, "text": text}


rows = []
with ThreadPoolExecutor(max_workers=1) as ex:  # concurrent proxy calls kill the interactive session
    for i, row in enumerate(ex.map(run, jobs), 1):
        rows.append(row)
        # a busy kernel that prints nothing for a while gets its session killed -> heartbeat
        print(f"{i} {row['model']} {len(row['text'].split())}w", flush=True)
        if i % 24 == 0 or i == len(jobs):  # checkpoint
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            rows = []
            print(f"{i}/{len(jobs)} saved", flush=True)

gen = pd.read_csv(OUT)
err = gen.text.str.startswith("ERROR")
print(f"total {len(gen)}, errors {err.sum()}")
if err.any():
    print(gen[err].groupby("model").text.first().str[:150].to_string())
t = gen[~err].text
print(f"words {t.str.split().str.len().mean():.0f} | newline {t.str.contains(chr(10)).mean():.0%} | "
      f"markdown {t.str.contains(r'[*#]', regex=True).mean():.0%}")
