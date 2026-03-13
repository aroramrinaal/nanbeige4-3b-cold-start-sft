import json
import os
from statistics import mean


def summarize_lengths(lengths):
    sorted_lengths = sorted(lengths)

    def percentile(p):
        if not sorted_lengths:
            return 0
        idx = min(len(sorted_lengths) - 1, round((p / 100) * (len(sorted_lengths) - 1)))
        return sorted_lengths[idx]

    return {
        "count": len(sorted_lengths),
        "min": min(sorted_lengths) if sorted_lengths else 0,
        "max": max(sorted_lengths) if sorted_lengths else 0,
        "mean": round(mean(sorted_lengths), 2) if sorted_lengths else 0,
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def analyze_token_lengths(dataset, tokenizer, cutoffs):
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id

    results = {
        "full_lengths": [],
        "prompt_lengths": [],
        "completion_lengths": [],
        "per_cutoff": {
            str(cutoff): {
                "full_truncated": 0,
                "completion_truncated": 0,
                "end_think_survives": 0,
                "eos_survives": 0,
            }
            for cutoff in cutoffs
        },
    }

    for ex in dataset:
        prompt = ex["prompt"]
        completion = ex["completion"]
        full_ids = tokenizer.encode(prompt + completion, add_special_tokens=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)

        results["full_lengths"].append(len(full_ids))
        results["prompt_lengths"].append(len(prompt_ids))
        results["completion_lengths"].append(len(completion_ids))

        end_think_index = max((i for i, tok in enumerate(full_ids) if tok == think_close_id), default=-1)
        eos_index = max((i for i, tok in enumerate(full_ids) if tok == eos_id), default=-1)

        for cutoff in cutoffs:
            bucket = results["per_cutoff"][str(cutoff)]
            if len(full_ids) > cutoff:
                bucket["full_truncated"] += 1
            if len(completion_ids) > cutoff:
                bucket["completion_truncated"] += 1
            if 0 <= end_think_index < cutoff:
                bucket["end_think_survives"] += 1
            if 0 <= eos_index < cutoff:
                bucket["eos_survives"] += 1

    total = len(dataset)
    return {
        "total_examples": total,
        "full_summary": summarize_lengths(results["full_lengths"]),
        "prompt_summary": summarize_lengths(results["prompt_lengths"]),
        "completion_summary": summarize_lengths(results["completion_lengths"]),
        "full_lengths": results["full_lengths"],
        "completion_lengths": results["completion_lengths"],
        "per_cutoff": {
            cutoff: {
                **stats,
                "full_truncated_pct": round(100 * stats["full_truncated"] / total, 2),
                "completion_truncated_pct": round(100 * stats["completion_truncated"] / total, 2),
                "end_think_survives_pct": round(100 * stats["end_think_survives"] / total, 2),
                "eos_survives_pct": round(100 * stats["eos_survives"] / total, 2),
            }
            for cutoff, stats in results["per_cutoff"].items()
        },
    }


def plot_full_sequence_histogram(full_lengths, cutoffs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(full_lengths, bins=40, color="#0f766e", edgecolor="white", linewidth=0.5)
    for cutoff in cutoffs:
        ax.axvline(
            cutoff,
            linestyle="--",
            linewidth=1.5,
            label=f"cutoff={cutoff}",
        )
    ax.set_xlabel("full sequence length in tokens")
    ax.set_ylabel("number of examples")
    ax.set_title("Opus full training sequence token lengths")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_sequence_token_histogram.png"), dpi=150)
    plt.close()


def plot_completion_histogram(completion_lengths, cutoffs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(completion_lengths, bins=40, color="#2563eb", edgecolor="white", linewidth=0.5)
    for cutoff in cutoffs:
        ax.axvline(
            cutoff,
            linestyle="--",
            linewidth=1.5,
            label=f"cutoff={cutoff}",
        )
    ax.set_xlabel("assistant completion length in tokens")
    ax.set_ylabel("number of examples")
    ax.set_title("Opus assistant-side completion token lengths")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "completion_token_histogram.png"), dpi=150)
    plt.close()


def plot_cutoff_tradeoffs(per_cutoff, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cutoffs = list(per_cutoff.keys())
    x = np.arange(len(cutoffs))
    width = 0.2

    full_truncated = [per_cutoff[c]["full_truncated_pct"] for c in cutoffs]
    completion_truncated = [per_cutoff[c]["completion_truncated_pct"] for c in cutoffs]
    end_think_survives = [per_cutoff[c]["end_think_survives_pct"] for c in cutoffs]
    eos_survives = [per_cutoff[c]["eos_survives_pct"] for c in cutoffs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, full_truncated, width, label="full seq truncated %", color="#dc2626")
    ax.bar(x - 0.5 * width, completion_truncated, width, label="completion-only > cutoff %", color="#f97316")
    ax.bar(x + 0.5 * width, end_think_survives, width, label="</think> survives %", color="#16a34a")
    ax.bar(x + 1.5 * width, eos_survives, width, label="</s> survives %", color="#0891b2")

    ax.set_xticks(x)
    ax.set_xticklabels(cutoffs)
    ax.set_xlabel("candidate max_length")
    ax.set_ylabel("percent of examples")
    ax.set_title("Cutoff tradeoffs for Opus token lengths")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cutoff_tradeoffs.png"), dpi=150)
    plt.close()


def write_report(analysis, output_dir):
    json_path = os.path.join(output_dir, "token_length_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    lines = [
        "Opus 4.6 token length analysis",
        "",
        f"total examples: {analysis['total_examples']}",
        "",
        "full sequence summary:",
        json.dumps(analysis["full_summary"], indent=2),
        "",
        "prompt summary:",
        json.dumps(analysis["prompt_summary"], indent=2),
        "",
        "completion summary:",
        json.dumps(analysis["completion_summary"], indent=2),
        "",
        "cutoff tradeoffs:",
    ]

    for cutoff, stats in analysis["per_cutoff"].items():
        lines.extend(
            [
                f"- cutoff {cutoff}",
                f"  full truncated: {stats['full_truncated']} ({stats['full_truncated_pct']}%)",
                f"  completion-only > cutoff: {stats['completion_truncated']} ({stats['completion_truncated_pct']}%)",
                f"  </think> survives: {stats['end_think_survives']} ({stats['end_think_survives_pct']}%)",
                f"  </s> survives: {stats['eos_survives']} ({stats['eos_survives_pct']}%)",
            ]
        )

    txt_path = os.path.join(output_dir, "token_length_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_all_token_graphs(analysis, output_dir, cutoffs):
    os.makedirs(output_dir, exist_ok=True)
    plot_full_sequence_histogram(analysis["full_lengths"], cutoffs, output_dir)
    plot_completion_histogram(analysis["completion_lengths"], cutoffs, output_dir)
    plot_cutoff_tradeoffs(analysis["per_cutoff"], output_dir)
    write_report(analysis, output_dir)
