"""Generate answers on the v2 test set with three local models: base, v1 adapter, v2 adapter.

Run from the finetune folder with ft-venv (GPU), after train_v2.py has finished:
    python eval_generate_v2.py
Takes roughly 20-30 minutes for ~150 test questions x 3 models on an RTX 3060.
Then run eval_judge_v2.py (project venv) to score them, together with gpt-4o-mini.
"""

import gc
import json
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

TEST_FILE = Path(r"C:\Users\preri\ai-study-assistant\finetune_data_v2\test.jsonl")
OUT_FILE = Path(r"C:\Users\preri\finetune\predictions_v2.jsonl")

MODELS = [
    ("base_answer", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "BASE model"),
    ("v1_answer", r"C:\Users\preri\finetune\lora_adapter", "FINE-TUNED v1"),
    ("v2_answer", r"C:\Users\preri\finetune\lora_adapter_v2", "FINE-TUNED v2"),
]

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 200


def load_test():
    with open(TEST_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_all(model_name, rows, label):
    print(f"\n=== Generating with {label} ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    answers, seconds = [], []
    for i, row in enumerate(rows, start=1):
        inputs = tokenizer.apply_chat_template(
            row["messages"][:2],  # system + user; the assistant answer is what we generate
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        input_len = inputs["input_ids"].shape[1]
        if input_len > MAX_SEQ_LENGTH - MAX_NEW_TOKENS:
            answers.append(None)
            seconds.append(None)
            continue

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        seconds.append(round(time.time() - start, 2))
        answers.append(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip())

        if i % 10 == 0:
            print(f"  {i}/{len(rows)} done")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return answers, seconds


def main():
    rows = load_test()
    print(f"Test examples: {len(rows)}")

    outputs = {}
    for key, name, label in MODELS:
        outputs[key] = generate_all(name, rows, label)

    saved = skipped = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if any(outputs[key][0][i] is None for key, _, _ in MODELS):
                skipped += 1
                continue
            record = {
                "paper": row["paper"],
                "type": row["type"],
                "gold": row["gold"],
                "question": row["question"],
                "reference": row["reference"],
                "prompt_messages": row["messages"][:2],
            }
            for key, _, _ in MODELS:
                record[key] = outputs[key][0][i]
                record[key.replace("_answer", "_seconds")] = outputs[key][1][i]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    print(f"\nSaved {saved} predictions (skipped {skipped} too-long prompts)")
    print(f"File: {OUT_FILE}")


if __name__ == "__main__":
    main()
