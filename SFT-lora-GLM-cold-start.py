# SFT with LoRA on Nanbeige4-3B-Base — COLD START on GLM-5.0 dataset
# dataset: crownelius/GLM-5.0-25000x (using 12k cleaned rows)
# runs on modal H100 GPU, saves lora adapter weights to modal volume
#
# base model tokenizer special tokens:
#   <s>      (id 1)       - bos, added automatically by tokenizer (add_bos_token: true)
#   </s>     (id 2)       - eos, NOT added automatically (add_eos_token: false)
#   <think>  (id 166103)  - in vocab natively, tokenized as a single token
#   </think> (id 166104)  - in vocab natively, tokenized as a single token
#
# FORMAT STRATEGY (native trl prompt-completion):
#
# prompt     = "User: {problem}\n"
# completion = "Assistant: <think>\n{thinking}\n</think>\n{solution}</s>"
#


import modal
import os

# -------------------------------------------------------------------
# modal volume — same volume as v1, different subfolder
# -------------------------------------------------------------------
volume = modal.Volume.from_name("nanbeige4-lora-output", create_if_missing=True)
VOLUME_MOUNT_PATH = "/mnt/lora-output"

app = modal.App("nanbeige4-3b-sft-lora")

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
)

MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"
DATASET_NAME = "crownelius/GLM-5.0-25000x"
OUTPUT_DIR = f"{VOLUME_MOUNT_PATH}/nanbeige4-3b-lora-GLM-5.0-12000x"
MAX_ROWS = 12_000
MIN_THINKING_CHARS = 20


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


def filter_empty_thinking(example):
    # drop rows where thinking trace is empty or too short to be useful
    # training on empty thinking teaches the model to skip reasoning
    return len(example["thinking"].strip()) >= MIN_THINKING_CHARS


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. load tokenizer
    # ----------------------------------------------------------------
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # sanity check: <think> and </think> must be single tokens
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    print(f"<think> token ids: {think_ids}")
    print(f"</think> token ids: {end_think_ids}")
    assert len(think_ids) == 1, f"<think> is not a single token: {think_ids}"
    assert len(end_think_ids) == 1, f"</think> is not a single token: {end_think_ids}"

    # ----------------------------------------------------------------
    # 2. load base model in bf16
    # ----------------------------------------------------------------
    print("loading base model...")
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
    # 3. lora config — same as v1
    # ----------------------------------------------------------------
    print("setting up lora config...")
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
    # 4. load, filter, cap, and format dataset
    # ----------------------------------------------------------------
    print(f"loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"raw dataset size: {len(dataset)} rows")

    before_filter = len(dataset)
    dataset = dataset.filter(filter_empty_thinking)
    after_filter = len(dataset)
    print(f"after filtering empty thinking (< {MIN_THINKING_CHARS} chars): "
          f"{after_filter} rows (removed {before_filter - after_filter})")

    if len(dataset) > MAX_ROWS:
        dataset = dataset.select(range(MAX_ROWS))
    print(f"final dataset size (capped at {MAX_ROWS}): {len(dataset)} rows")

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    print("sample prompt:")
    print(dataset[0]["prompt"])
    print("sample completion (first 300 chars):")
    print(dataset[0]["completion"][:300])
    print("---")

    # token length histogram — runs before training
    # tells you if max_length=2048 is going to truncate many examples
    plot_all_graphs(
        log_history=None,
        output_dir=OUTPUT_DIR,
        dataset=dataset,
        tokenizer=tokenizer,
        pre_training_only=True,
    )

    # ----------------------------------------------------------------
    # 5. training config
    # ----------------------------------------------------------------
    print("setting up training args...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch = 16
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
        # do NOT set dataset_text_field — using prompt-completion format
    )

    # ----------------------------------------------------------------
    # 6. trainer
    # ----------------------------------------------------------------
    print("setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ----------------------------------------------------------------
    # 7. train
    # ----------------------------------------------------------------
    print("starting training...")
    trainer.train()

    # ----------------------------------------------------------------
    # 8. post-training graphs (loss curve, grad norm, per-epoch bars)
    # ----------------------------------------------------------------
    log_history = trainer.state.log_history

    plot_all_graphs(
        log_history=log_history,
        output_dir=OUTPUT_DIR,
        dataset=None,
        tokenizer=None,
        pre_training_only=False,
    )

    loss_entries = [x for x in log_history if "loss" in x]
    if loss_entries:
        print(f"\n--- training summary ---")
        print(f"initial loss: {loss_entries[0]['loss']:.4f}")
        print(f"final loss:   {loss_entries[-1]['loss']:.4f}")
        print(f"total drop:   {loss_entries[0]['loss'] - loss_entries[-1]['loss']:.4f}")
        print(f"------------------------\n")

    # ----------------------------------------------------------------
    # 9. save lora adapter weights + tokenizer
    # ----------------------------------------------------------------
    print(f"saving lora adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    volume.commit()

    print("done! files saved to modal volume.")
    print(f"volume: nanbeige4-lora-output  path: {OUTPUT_DIR}")
    for f in os.listdir(OUTPUT_DIR):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"  {f}  ({size_kb} KB)")


@app.local_entrypoint()
def main():
    run_sft_training.remote()
    print("done! training completed.")
