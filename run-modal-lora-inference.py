import os

import modal


APP_NAME = "nanbeige4-3b-lora-inference"
MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"
VOLUME_NAME = "nanbeige4-lora-output"
VOLUME_MOUNT_PATH = "/mnt/lora-output"
ADAPTER_SUBDIR = "nanbeige4-3b-lora-v1"
DEFAULT_MAX_NEW_TOKENS = 256

HELD_OUT_PROMPTS = [
    "If 5 workers finish a task in 12 days at the same rate, how many days would 8 workers need? Show the reasoning clearly.",
    "A shop gives 20% off and then an extra 10% off. Is that the same as 30% off? Explain with an example price of $100.",
    "Mira has 3 red marbles, 5 blue marbles, and 2 green marbles. If she picks one marble at random, what is the probability it is not blue?",
]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "sentencepiece",
        "safetensors",
    )
)


def build_prompt(user_prompt: str) -> str:
    # Match the SFT prompt format exactly so generation starts with the learned
    # completion pattern: "Assistant: <think> ..."
    return f"User: {user_prompt.strip()}\n"


@app.function(
    image=image,
    gpu="A10",
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_lora_inference(
    prompts: list[str],
    adapter_subdir: str = ADAPTER_SUBDIR,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ["HF_TOKEN"]
    adapter_path = f"{VOLUME_MOUNT_PATH}/{adapter_subdir}"

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model from {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    results = []
    for idx, prompt in enumerate(prompts, start=1):
        formatted_prompt = build_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        result = {
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "response": generated_text,
        }
        results.append(result)

        print(f"\n=== Prompt {idx} ===")
        print(prompt)
        print("\n--- Model output ---")
        print(generated_text)

    return results


@app.local_entrypoint()
def main(
    adapter_subdir: str = ADAPTER_SUBDIR,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    results = run_lora_inference.remote(
        prompts=HELD_OUT_PROMPTS,
        adapter_subdir=adapter_subdir,
        max_new_tokens=max_new_tokens,
    )

    print("\n=== Summary ===")
    for idx, result in enumerate(results, start=1):
        print(f"\nPrompt {idx}: {result['prompt']}")
        print(result["response"])
