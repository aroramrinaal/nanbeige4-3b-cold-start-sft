# training_graphs.py
#
# all plotting logic lives here, imported by SFT-lora-GLM-cold-start.py
#
# called twice during a training run:
#   1. before training  (pre_training_only=True)  -> token length histogram only
#   2. after training   (pre_training_only=False) -> loss curve, grad norm, per-epoch bars
#
# output: PNG files saved to OUTPUT_DIR inside the modal volume
# you can download them directly from the modal UI after the run finishes
#
# -----------------------------------------------------------------------
# WHAT EACH GRAPH TELLS YOU (student researcher notes)
# -----------------------------------------------------------------------
#
# token_length_histogram.png  (generated BEFORE training)
#   - shows the distribution of how many tokens each example uses
#   - the red dashed line is your max_length=2048 cutoff
#   - any bar to the right of that line = examples that will be truncated
#   - truncation means the model never sees the end of those reasoning traces
#   - if > 20-30% of your data is being truncated, consider raising max_length
#     or filtering out very long examples
#   - if everything is short (< 500 tokens), training will be fast but the model
#     won't learn long-form multi-step reasoning
#
# training_loss_curve.png  (generated AFTER training)
#   - the most important graph to look at
#   - x-axis = training steps, y-axis = loss value logged every 10 steps
#   - healthy shape: drops fast in first ~20% of training, then flattens out
#   - still falling steeply at the end = model could benefit from more epochs
#   - completely flat from early on = model converged early, extra compute wasted,
#     or your learning rate is too low and the model stopped making progress
#   - very spiky/noisy throughout = lr might be too high, or batch size too small
#   - the start and end loss values are annotated directly on the plot
#
# loss_per_epoch.png  (generated AFTER training)
#   - two bars: average loss in epoch 1 vs epoch 2
#   - epoch 2 should always be clearly lower than epoch 1
#   - large drop = the model had a lot left to learn, could benefit from epoch 3
#   - small drop = mostly converged in epoch 1, epoch 2 was fine-tuning
#   - epoch 2 higher than epoch 1 = something is wrong (data quality issue,
#     lr too high causing instability, or the model is overfitting)
#
# grad_norm_curve.png  (generated AFTER training)
#   - gradient norm = the size of the parameter update signal at each step
#   - stable training = norms stay roughly consistent, maybe drifting slightly down
#   - sudden large spikes = model hit an outlier batch (very long/unusual example),
#     or the lr is too aggressive for some batches
#   - consistently high norms (> 5-10) with no convergence = lr might be too high
#   - very flat low norms throughout = the model has stopped learning
#   - the average norm is shown as a dashed red line for reference
#   - note: SFTConfig clips gradients at max_grad_norm=1.0 by default, so you
#     won't usually see norms above 1.0 unless you changed that setting
# -----------------------------------------------------------------------

import os
import math


def plot_token_length_histogram(dataset, tokenizer, output_dir, sample_size=500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random

    print(f"computing token lengths on {sample_size} random examples...")
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    lengths = []
    for i in indices:
        ex = dataset[i]
        full_text = ex["prompt"] + ex["completion"]
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        lengths.append(len(tokens))

    truncated = sum(1 for l in lengths if l > 2048)
    truncated_pct = 100 * truncated / len(lengths)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=40, color="#0891b2", edgecolor="white", linewidth=0.5)
    ax.axvline(2048, color="#dc2626", linestyle="--", linewidth=1.5,
               label="max_length=2048 (truncation point)")
    ax.set_xlabel("token length", fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    ax.set_title(
        f"token length distribution — {sample_size} random examples\n"
        f"truncated (> 2048): {truncated} / {len(lengths)} examples ({truncated_pct:.1f}%)",
        fontsize=11
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "token_length_histogram.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved: {save_path}")

    print(f"token length stats:")
    print(f"  min:    {min(lengths)}")
    print(f"  max:    {max(lengths)}")
    print(f"  mean:   {sum(lengths)/len(lengths):.1f}")
    print(f"  > 2048: {truncated} / {len(lengths)} ({truncated_pct:.1f}%)")


def plot_training_loss(log_history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [x["step"] for x in log_history if "loss" in x]
    losses = [x["loss"] for x in log_history if "loss" in x]

    if not steps:
        print("no loss data in log_history, skipping loss curve")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=1.5, color="#2563eb", alpha=0.8)
    ax.set_xlabel("training step", fontsize=12)
    ax.set_ylabel("loss", fontsize=12)
    ax.set_title("training loss over steps", fontsize=11)
    ax.grid(True, alpha=0.3)

    if len(losses) >= 2:
        ax.annotate(f"start: {losses[0]:.3f}", xy=(steps[0], losses[0]),
                    xytext=(steps[0], losses[0] + 0.1),
                    fontsize=9, color="#dc2626")
        ax.annotate(f"end: {losses[-1]:.3f}", xy=(steps[-1], losses[-1]),
                    xytext=(steps[-1] * 0.85, losses[-1] + 0.1),
                    fontsize=9, color="#16a34a")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved: {save_path}")


def plot_loss_per_epoch(log_history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    epoch_losses = {}
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            ep = max(1, math.ceil(entry["epoch"]))
            epoch_losses.setdefault(ep, []).append(entry["loss"])

    if not epoch_losses:
        print("no epoch loss data in log_history, skipping per-epoch plot")
        return

    epochs = sorted(epoch_losses.keys())
    avg_losses = [np.mean(epoch_losses[e]) for e in epochs]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [str(e) for e in epochs],
        avg_losses,
        color=["#3b82f6", "#10b981"][:len(epochs)],
        width=0.5
    )
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel("average loss", fontsize=12)
    ax.set_title("average training loss per epoch", fontsize=11)

    for bar, val in zip(bars, avg_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max(avg_losses) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "loss_per_epoch.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved: {save_path}")


def plot_grad_norm(log_history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [x["step"] for x in log_history if "grad_norm" in x]
    norms = [x["grad_norm"] for x in log_history if "grad_norm" in x]

    if not steps:
        print("no grad_norm data in log_history, skipping grad norm plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, norms, linewidth=1.2, color="#7c3aed", alpha=0.8)
    ax.set_xlabel("training step", fontsize=12)
    ax.set_ylabel("gradient norm", fontsize=12)
    ax.set_title("gradient norm over training", fontsize=11)
    ax.grid(True, alpha=0.3)

    avg_norm = sum(norms) / len(norms)
    ax.axhline(avg_norm, color="#dc2626", linestyle="--", alpha=0.6,
               label=f"avg: {avg_norm:.3f}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "grad_norm_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved: {save_path}")


def plot_all_graphs(log_history, output_dir, dataset, tokenizer,
                    pre_training_only=False):
    """
    single entry point called from the training script.

    pre_training_only=True  -> only runs token length histogram (before training)
    pre_training_only=False -> only runs loss/grad norm graphs (after training)
    """
    if pre_training_only:
        print("generating pre-training graph: token length histogram...")
        plot_token_length_histogram(dataset, tokenizer, output_dir, sample_size=500)
    else:
        print("generating post-training graphs...")
        plot_training_loss(log_history, output_dir)
        plot_loss_per_epoch(log_history, output_dir)
        plot_grad_norm(log_history, output_dir)
        print("all graphs saved.")
