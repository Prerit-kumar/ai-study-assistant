import torch
from unsloth import FastLanguageModel


def main():
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

    print("Loading model (first time download ~2GB)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    used = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM used after loading: {used:.2f} GB")

    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "user", "content": "Explain what RAG is in 2 simple sentences."}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    print("\n--- Model answer ---")
    print(answer)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak VRAM: {peak:.2f} GB")


if __name__ == "__main__":
    main()