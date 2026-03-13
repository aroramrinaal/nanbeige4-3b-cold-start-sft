# SFT with LoRA on Nanbeige4-3B-Base
# dataset: crownelius/Opus-4.6-Reasoning-3300x (2160 rows)
# runs on modal H100 GPU, saves lora adapter weights to modal volume
#
# KEY DESIGN DECISIONS (based on tokenizer_config.json of the base model):
#
# base model tokenizer special tokens:
#   <s>      (id 1)       - bos, added automatically by tokenizer (add_bos_token: true)
#   </s>     (id 2)       - eos, NOT added automatically (add_eos_token: false)
#   <think>  (id 166103)  - in vocab natively, tokenized as a single token
#   </think> (id 166104)  - in vocab natively, tokenized as a single token
#
# FORMAT STRATEGY (native trl prompt-completion):
#
# we use trl's native {"prompt": ..., "completion": ...} dataset format.
# per trl docs: "By default, the trainer computes the loss on the completion
# tokens only, ignoring the prompt tokens."
# this is cleaner than DataCollatorForCompletionOnlyLM which does string
# matching on tokenized output and can silently fail if context changes tokenization.
#
# prompt     = "User: {problem}\n"
# completion = "Assistant: <think>\n{thinking}\n</think>\n{solution}</s>"
#
# what the model is trained on (loss computed here):
#   Assistant: <think>\n   <- model learns to emit the assistant wrapper + opening think tag
#   {thinking}             <- learns the reasoning trace format
#   </think>\n             <- learns to close thinking
#   {solution}             <- learns to produce the final answer
#   </s>                   <- learns to stop
#
# the tokenizer auto-adds <s> at the front of the prompt (add_bos_token: true).
# we manually add </s> at the end of completion (add_eos_token: false).

import modal
import os

# -------------------------------------------------------------------
# modal volume to persist the lora adapter weights after training
# -------------------------------------------------------------------
volume = modal.Volume.from_name("nanbeige4-lora-output", create_if_missing=True)
VOLUME_MOUNT_PATH = "/mnt/lora-output"

app = modal.App("nanbeige4-3b-sft-lora-opus-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "trl>=0.29.0",
        "peft",
        "sentencepiece",
        "safetensors",
        "matplotlib",
        "numpy",
    )
    .add_local_python_source("training_graphs")
)

MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"
DATASET_NAME = "crownelius/Opus-4.6-Reasoning-3300x"
OUTPUT_DIR = f"{VOLUME_MOUNT_PATH}/opus-4.6-experiment-v2/nanbeige4-3b-lora-v2-epoch3"


def format_example(example):
    problem = example["problem"].strip()
    thinking = example["thinking"].strip()
    solution = example["solution"].strip()

    # native trl prompt-completion format
    # trl will mask the prompt tokens and compute loss on completion only
    # tokenizer auto-adds <s> before prompt, so don't add it manually
    # </s> is added manually at end of completion since add_eos_token is false
    return {
        "prompt": (
            f"User: {problem}\n"
        ),
        "completion": (
            f"Assistant: <think>\n"
            f"{thinking}\n</think>\n"
            f"{solution}</s>"
        ),
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_sft_training():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from training_graphs import plot_all_graphs

    hf_token = os.environ["HF_TOKEN"]

    # ----------------------------------------------------------------
    # 1. load tokenizer
    # ----------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )

    # LlamaTokenizer has no pad token by default, set it to eos
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # sanity checks: ensure <think> and </think> are each single tokens
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    print(f"<think> token ids: {think_ids}")       # actual ids may differ from hub config
    print(f"</think> token ids: {end_think_ids}")  # runtime tokenizer state is source of truth
    assert len(think_ids) == 1, f"<think> is not a single token: {think_ids}"
    assert len(end_think_ids) == 1, f"</think> is not a single token: {end_think_ids}"

    # ----------------------------------------------------------------
    # 2. load base model in bf16
    # ----------------------------------------------------------------
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ----------------------------------------------------------------
    # 3. lora config
    # ----------------------------------------------------------------
    print("Setting up lora config...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # expect: trainable params ~13M / all params ~3B / trainable% ~0.4%

    # ----------------------------------------------------------------
    # 4. load and format dataset
    # ----------------------------------------------------------------
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"dataset size: {len(dataset)} rows")

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print("Sample prompt:")
    print(dataset[0]["prompt"])
    print("sample completion (first 300 chars):")
    print(dataset[0]["completion"][:300])
    print("---")

    plot_all_graphs(
        log_history=[],
        output_dir=OUTPUT_DIR,
        dataset=dataset,
        tokenizer=tokenizer,
        pre_training_only=True,
    )

    # ----------------------------------------------------------------
    # 5. training config
    #
    # no DataCollatorForCompletionOnlyLM needed.
    # trl's native prompt-completion format handles loss masking automatically.
    # the prompt tokens get labels = -100, completion tokens get normal loss.
    # ----------------------------------------------------------------
    print("Setting up training args...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,   # h100 80gb handles 3b + lora at batch 4 fine
        gradient_accumulation_steps=4,   # effective batch size = 4 * 4 = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=2048,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        # do NOT set dataset_text_field here since we are using prompt-completion format
    )

    # ----------------------------------------------------------------
    # 6. trainer
    # ----------------------------------------------------------------
    print("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        # no data_collator needed, trl handles completion-only loss natively
    )

    # ----------------------------------------------------------------
    # 7. train
    # ----------------------------------------------------------------
    print("starting training...")
    trainer.train()

    plot_all_graphs(
        log_history=trainer.state.log_history,
        output_dir=OUTPUT_DIR,
        dataset=dataset,
        tokenizer=tokenizer,
        pre_training_only=False,
    )

    # ----------------------------------------------------------------
    # 8. save lora adapter weights to modal volume
    # saves only adapter_config.json + adapter_model.safetensors (~50-150MB)
    # ----------------------------------------------------------------
    print(f"Saving lora adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    volume.commit()

    print("Done! adapter weights saved.")
    print(f"Find your files in modal volume: nanbeige4-lora-output at path: {OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    run_sft_training.remote()
    print("Done! Training completed.")
