"""Generate answers with fine-tuned v2 and v3 on two test sets:
  docs   -> finetune_data_v3/test.jsonl  (30 unseen everyday documents, 4 document types never trained on)
  papers -> finetune_data_v2/test.jsonl  (the old research-paper test: did v3 forget papers?)

Run from the finetune folder with ft-venv (GPU), after train_v3.py has finished:
    python C:\\Users\\preri\\ai-study-assistant\\finetune\\eval_generate_v3.py
Takes roughly 30-40 minutes on an RTX 3060. Then run eval_judge_v3.py (project venv) to score them.
"""

import gc
import json
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

TEST_SETS = {
    "docs": Path(r"C:\Users\preri\ai-study-assistant\finetune_data_v3\test.jsonl"),
    "papers": Path(r"C:\Users\preri\ai-study-assistant\finetune_data_v2\test.jsonl"),
}
OUT_DIR = Path(r"C:\Users\preri\finetune")

MODELS = [
    ("v2_answer", r"C:\Users\preri\finetune\lora_adapter_v2", "FINE-TUNED v2"),
    ("v3_answer", r"C:\Users\preri\finetune\lora_adapter_v3", "FINE-TUNED v3"),
]

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 200


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_all(model, tokenizer, rows, label):
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
        if i % 25 == 0:
            print(f"  {label}: {i}/{len(rows)}")
    return answers, seconds


def main():
    sets = {name: load(path) for name, path in TEST_SETS.items()}
    for name, rows in sets.items():
        print(f"{name}: {len(rows)} test questions")

    outputs = {name: {} for name in sets}
    for key, adapter, label in MODELS:
        print(f"\n=== Loading {label} ===")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        for name, rows in sets.items():
            outputs[name][key] = generate_all(model, tokenizer, rows, f"{label} / {name}")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    for name, rows in sets.items():
        out_file = OUT_DIR / f"predictions_v3_{name}.jsonl"
        saved = skipped = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                if any(outputs[name][key][0][i] is None for key, _, _ in MODELS):
                    skipped += 1
                    continue
                record = {
                    "paper": row["paper"],
                    "doc_type": row.get("doc_type", "research paper"),
                    "ocr": row.get("ocr", False),
                    "type": row["type"],
                    "gold": row["gold"],
                    "question": row["question"],
                    "reference": row["reference"],
                    "prompt_messages": row["messages"][:2],
                }
                for key, _, _ in MODELS:
                    record[key] = outputs[name][key][0][i]
                    record[key.replace("_answer", "_seconds")] = outputs[name][key][1][i]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
        print(f"{name}: saved {saved} predictions (skipped {skipped} too-long prompts) -> {out_file}")


if __name__ == "__main__":
    main()
