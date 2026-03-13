import os
import sys

import modal


volume = modal.Volume.from_name("nanbeige4-lora-output", create_if_missing=True)
VOLUME_MOUNT_PATH = "/mnt/lora-output"
MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"
DATASET_NAME = "crownelius/Opus-4.6-Reasoning-3300x"
OUTPUT_DIR = f"{VOLUME_MOUNT_PATH}/opus-4.6-experiment-v2/token-analysis-v1"
CUTOFFS = (2048, 3072, 4096)

app = modal.App("nanbeige4-opus-token-analysis")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "matplotlib",
        "numpy",
        "sentencepiece",
        "transformers",
    )
    .add_local_dir(
        "Opus-4.6-experiment-v2",
        remote_path="/root/Opus-4.6-experiment-v2",
    )
)


def format_example(example):
    problem = example["problem"].strip()
    thinking = example["thinking"].strip()
    solution = example["solution"].strip()
    return {
        "prompt": f"User: {problem}\n",
        "completion": (
            f"Assistant: <think>\n"
            f"{thinking}\n</think>\n"
            f"{solution}</s>"
        ),
    }


def recommend_next_max_length(per_cutoff):
    candidates = []
    for cutoff, stats in per_cutoff.items():
        candidates.append(
            {
                "cutoff": int(cutoff),
                "full_truncated_pct": stats["full_truncated_pct"],
                "eos_survives_pct": stats["eos_survives_pct"],
                "end_think_survives_pct": stats["end_think_survives_pct"],
            }
        )

    candidates.sort(key=lambda item: item["cutoff"])

    for candidate in candidates:
        if candidate["full_truncated_pct"] <= 10 and candidate["eos_survives_pct"] >= 90:
            return (
                f"recommended next max_length: {candidate['cutoff']} "
                f"(full truncation {candidate['full_truncated_pct']}%, "
                f"</s> survives {candidate['eos_survives_pct']}%)"
            )

    best = max(
        candidates,
        key=lambda item: (item["eos_survives_pct"], item["end_think_survives_pct"], -item["full_truncated_pct"]),
    )
    return (
        f"no tested cutoff cleanly meets the target, but the strongest candidate is {best['cutoff']} "
        f"(full truncation {best['full_truncated_pct']}%, "
        f"</s> survives {best['eos_survives_pct']}%)"
    )


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_token_analysis():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    sys.path.append("/root/Opus-4.6-experiment-v2")
    from token_graphs import analyze_token_lengths, save_all_token_graphs

    hf_token = os.environ["HF_TOKEN"]

    print("Loading tokenizer only on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Opus dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", token=hf_token)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    print(f"Running token analysis on all {len(dataset)} rows...")
    analysis = analyze_token_lengths(dataset, tokenizer, CUTOFFS)
    save_all_token_graphs(analysis, OUTPUT_DIR, CUTOFFS)
    recommendation = recommend_next_max_length(analysis["per_cutoff"])

    volume.commit()

    print("Token analysis complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Full sequence summary:")
    print(analysis["full_summary"])
    print("Completion summary:")
    print(analysis["completion_summary"])
    print("Cutoff tradeoffs:")
    for cutoff, stats in analysis["per_cutoff"].items():
        print(cutoff, stats)
    print(recommendation)


@app.local_entrypoint()
def main():
    run_token_analysis.remote()
