# Opus 4.6 Experiment V2

This folder is for the Opus-only analysis and follow-up training runs.

## Files

- `analyze_opus_tokens_modal.py`: Modal CPU job that loads the tokenizer, formats the Opus dataset the same way as training, and saves token-length graphs plus reports to the Modal volume.
- `token_graphs.py`: shared graphing and token-analysis helpers used by the Modal analysis job.

## Output path on the Modal volume

- `/mnt/lora-output/opus-4.6-experiment-v2/token-analysis-v1`

## What this analysis measures

- full training sequence length: `<s> + prompt + completion`
- prompt length
- completion-only length
- cutoff tradeoffs for `2048`, `3072`, and `4096`
- whether `</think>` and `</s>` survive under each candidate cutoff
- prints a one-line recommended next `max_length` hint from the cutoff stats after the Modal CPU run
