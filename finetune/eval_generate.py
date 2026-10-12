import gc
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel

TEST_FILE = Path(r"C:\Users\preri\ai-study-assistant\finetune_data\test.jsonl")
OUT_FILE = Path(r"C:\Users\preri\finetune\predictions.jsonl")

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
ADAPTER_DIR = r"C:\Users\preri\finetune\lora_adapter"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 200


def load_test():
    rows = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate_all(model_name, rows, label):
    print(f"\n=== Generating with {label} ===")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    answers = []
    for i, row in enumerate(rows, start=1):
        # system + user only, the assistant answer is what we generate
        prompt_messages = row["messages"][:2]

        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        input_len = inputs["input_ids"].shape[1]
        if input_len > MAX_SEQ_LENGTH - MAX_NEW_TOKENS:
            answers.append(None)
            continue

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        text = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        answers.append(text)

        if i % 10 == 0:
            print(f"  {i}/{len(rows)} done")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def main():
    rows = load_test()
    print(f"Test examples: {len(rows)}")

    base_answers = generate_all(BASE_MODEL, rows, "BASE model")
    ft_answers = generate_all(ADAPTER_DIR, rows, "FINE-TUNED model")

    saved = 0
    skipped = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row, base, ft in zip(rows, base_answers, ft_answers):
            if base is None or ft is None:
                skipped += 1
                continue

            record = {
                "paper": row["paper"],
                "type": row["type"],
                "question": row["messages"][1]["content"]
                .split("Question:")[-1]
                .strip(),
                "reference": row["messages"][2]["content"],
                "base_answer": base,
                "finetuned_answer": ft,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    print(f"\nSaved {saved} predictions (skipped {skipped} too-long prompts)")
    print(f"File: {OUT_FILE}")


if __name__ == "__main__":
    main()