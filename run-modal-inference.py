
# first try to 
# load the model weights and run the inference on modal

import modal
import os

app = modal.App("nanbeige4-3b-base-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
    )
)

MODEL_NAME = "Nanbeige/Nanbeige4-3B-Base"

@app.function(
    image=image,
    gpu="A10",
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_inference(prompt: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ["HF_TOKEN"]

    print(f"loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
        token=hf_token,
    )

    print(f"loading model for {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
)

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(f"response: {response}")
    return response


@app.local_entrypoint()
def main():
    prompt = "The capital of India is"

    result = run_inference.remote(prompt)
    print("\n--- output ---")
    print(result)
