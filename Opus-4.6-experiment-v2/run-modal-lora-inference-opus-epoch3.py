import os

import modal


APP_NAME = "nanbeige4-3b-lora-inference-opus-epoch3"
MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"
VOLUME_NAME = "nanbeige4-lora-output"
VOLUME_MOUNT_PATH = "/mnt/lora-output"
ADAPTER_SUBDIR = "opus-4.6-experiment-v2/nanbeige4-3b-lora-v2-epoch3"
DEFAULT_MAX_NEW_TOKENS = 1024

TEST_PROMPT = "If 5 workers finish a task in 12 days at the same rate, how many days would 8 workers need? Show the reasoning clearly."

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
    return f"User: {user_prompt.strip()}\n"


def normalize_response(generated_text: str) -> str:
    text = generated_text.lstrip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:"):].lstrip()
    return text


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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    results = []
    for prompt in prompts:
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
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        normalized_text = normalize_response(generated_text)

        results.append(
            {
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "response": normalized_text,
                "raw_response": generated_text,
            }
        )

    return results


@app.local_entrypoint()
def main(
    adapter_subdir: str = ADAPTER_SUBDIR,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    result = run_lora_inference.remote(
        prompts=[TEST_PROMPT],
        adapter_subdir=adapter_subdir,
        max_new_tokens=max_new_tokens,
    )[0]

    print("\n" + "=" * 60)
    print("PROMPT")
    print("=" * 60)
    print(result["prompt"])
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result["response"])
