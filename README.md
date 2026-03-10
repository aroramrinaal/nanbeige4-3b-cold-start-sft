# nanbeige4-3b-cold-start-sft

LoRA cold-start SFT experiments teaching Nanbeige4-3B-Base to produce structured
`<think>` reasoning traces, using knowledge distillation from frontier models.

## Runs

| Run | Dataset | Rows | Adapter |
|-----|---------|------|---------|
| v1 | Opus 4.6 reasoning traces | 2,160 | [HF](https://huggingface.co/mrinaalarora/Nanbeige4-3B-Cold-Start-Reasoning-LoRA) |
| v2 | GLM 5.0 reasoning traces | 12,000 | [HF](https://huggingface.co/mrinaalarora/nanbeige4-3b-cold-start-reasoning-lora-glm-12k) |

## Format
```
User: {problem}
Assistant: <think>
{reasoning trace}
</think>
{solution}
```

## Stack
- base model: [Nanbeige/Nanbeige4-3B-Base](https://huggingface.co/Nanbeige/Nanbeige4-3B-Base)
- training: modal H100, trl SFTTrainer, peft LoRA r=16
- all adapter weights on [huggingface](https://huggingface.co/collections/mrinaalarora/nanbeige4-3b-cold-start-reasoning-lora-experiments)
