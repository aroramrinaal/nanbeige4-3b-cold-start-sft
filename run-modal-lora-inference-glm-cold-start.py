import os

import modal


APP_NAME = "nanbeige4-3b-lora-inference-glm-cold-start"
MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"
VOLUME_NAME = "nanbeige4-lora-output"
VOLUME_MOUNT_PATH = "/mnt/lora-output"
ADAPTER_SUBDIR = "nanbeige4-3b-lora-GLM-5.0-12000x"
DEFAULT_MAX_NEW_TOKENS = 1024

TEST_PROMPT = (
    "A store increases a price of $80 by 15%. What is the new price?"
)

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


def trim_repeated_completion(text: str) -> str:
    # Keep only the first completed reasoning/answer block if the model starts
    # another cycle like "</think> ... </think> ..." after the answer.
    first_close = text.find("</think>")
    if first_close == -1:
        return text

    second_close = text.find("</think>", first_close + len("</think>"))
    if second_close != -1:
        return text[:second_close].rstrip()

    repeated_think = text.find("<think>", first_close + len("</think>"))
    if repeated_think != -1:
        return text[:repeated_think].rstrip()

    return text.rstrip()


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
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

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
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    class RepeatThinkStop(StoppingCriteria):
        def __init__(self, tokenizer, prompt_lengths):
            self.tokenizer = tokenizer
            self.prompt_lengths = prompt_lengths

        def __call__(self, input_ids, scores, **kwargs):
            for row_idx in range(input_ids.shape[0]):
                generated_ids = input_ids[row_idx][self.prompt_lengths[row_idx]:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                if text.count("</think>") >= 2:
                    return True
                first_close = text.find("</think>")
                if first_close != -1 and text.find("<think>", first_close + len("</think>")) != -1:
                    return True
            return False

    results = []
    for prompt in prompts:
        formatted_prompt = build_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        stopping_criteria = StoppingCriteriaList(
            [RepeatThinkStop(tokenizer, [inputs["input_ids"].shape[1]])]
        )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        normalized_text = normalize_response(generated_text)
        cleaned_text = trim_repeated_completion(normalized_text)

        results.append(
            {
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "response": cleaned_text,
                "normalized_response": normalized_text,
                "raw_response": generated_text,
            }
        )

    return results


@app.local_entrypoint()
def main(
    adapter_subdir: str = ADAPTER_SUBDIR,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    prompt: str = TEST_PROMPT,
):
    results = run_lora_inference.remote(
        prompts=[prompt],
        adapter_subdir=adapter_subdir,
        max_new_tokens=max_new_tokens,
    )

    print("\n=== Summary ===")
    for idx, result in enumerate(results, start=1):
        print(f"\nPrompt {idx}: {result['prompt']}")
        print(result["response"])
