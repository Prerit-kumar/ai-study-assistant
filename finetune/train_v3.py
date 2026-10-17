import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_FILE = Path(r"C:\Users\preri\ai-study-assistant\finetune_data_v3\train.jsonl")
OUTPUT_DIR = Path(r"C:\Users\preri\finetune\outputs_v3")
ADAPTER_DIR = Path(r"C:\Users\preri\finetune\lora_adapter_v3")

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

MAX_SEQ_LENGTH = 2048
SEED = 42

# Dataset and training
VALIDATION_SIZE = 0.10  # fraction of PAPERS held out for validation
NUM_EPOCHS = 1  # v3: about 3,000 examples (new documents + research papers), about 2 hours
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATASET VALIDATION AND FORMATTING
# ============================================================

def validate_dataset(dataset):
    if len(dataset) < 2:
        raise ValueError("At least 2 examples are required.")

    if "paper" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'paper' column.")

    for index, example in enumerate(dataset):
        messages = example.get("messages")

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Example {index} must contain a non-empty messages list."
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Invalid message format in example {index}.")

            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported role in example {index}: {message.get('role')}"
                )

            if not isinstance(message.get("content"), str):
                raise ValueError(
                    f"Message content must be a string in example {index}."
                )

        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError(f"Example {index} has no assistant response.")


def format_example(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def prepare_datasets(tokenizer):
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    dataset = load_dataset(
        "json",
        data_files=str(TRAIN_FILE),
        split="train",
    )

    validate_dataset(dataset)

    print(f"Total examples: {len(dataset)}")

    # Split by paper so validation chunks are never seen during training.
    papers = sorted(set(dataset["paper"]))
    rng = random.Random(SEED)
    rng.shuffle(papers)

    num_val_papers = max(1, round(len(papers) * VALIDATION_SIZE))
    val_papers = set(papers[:num_val_papers])
    print(f"Validation papers: {sorted(val_papers)}")

    train_dataset = dataset.filter(lambda ex: ex["paper"] not in val_papers)
    eval_dataset = dataset.filter(lambda ex: ex["paper"] in val_papers)

    # Convert messages into the model's native chat format.
    train_dataset = train_dataset.map(
        lambda example: format_example(example, tokenizer),
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )

    eval_dataset = eval_dataset.map(
        lambda example: format_example(example, tokenizer),
        remove_columns=eval_dataset.column_names,
        num_proc=1,
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    # Inspect one example before training.
    print("\n========== FORMATTED SAMPLE ==========")
    print(train_dataset[0]["text"][:2000])
    print("======================================\n")

    return train_dataset, eval_dataset


# ============================================================
# 4. MAIN TRAINING PIPELINE
# ============================================================

def main():
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"GPU VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # --------------------------------------------------------
    # Load quantized base model
    # --------------------------------------------------------

    print("\nLoading base model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Attach LoRA adapters
    # --------------------------------------------------------

    print("Preparing LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    model.config.use_cache = False
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------------

    train_dataset, eval_dataset = prepare_datasets(tokenizer)

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    use_bf16 = is_bfloat16_supported()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        # Dataset
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_num_proc=1,

        # Batch and optimization
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=10,
        weight_decay=0.01,
        max_grad_norm=0.3,

        # Precision
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",

        # Scheduler
        lr_scheduler_type="linear",

        # Evaluation and checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="none",

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # Safer defaults for Windows
        dataloader_num_workers=0,
    )

    # --------------------------------------------------------
    # Initialize trainer
    # --------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Calculate loss only on assistant responses.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # --------------------------------------------------------
    # Verify response masking before the full training run
    # --------------------------------------------------------

    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    active_labels = (labels != -100).sum().item()

    print(f"\nTokens contributing to loss: {active_labels}")

    if active_labels == 0:
        raise RuntimeError(
            "No active labels found. Check chat template and response masking."
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting fine-tuning...")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    train_result = trainer.train()

    elapsed_minutes = (time.time() - start_time) / 60

    eval_metrics = trainer.evaluate()

    print("\n========== TRAINING RESULTS ==========")
    print(f"Training time: {elapsed_minutes:.2f} minutes")
    print(f"Training loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"Validation loss: {eval_metrics.get('eval_loss', 'N/A')}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak allocated VRAM: {peak_vram:.2f} GB")

    # --------------------------------------------------------
    # Save final adapter and tokenizer
    # --------------------------------------------------------

    print("\nSaving final adapter...")

    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    results = {
        "base_model": MODEL_NAME,
        "training_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),
        "epochs_requested": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_R,
        "max_seq_length": MAX_SEQ_LENGTH,
        "training_time_minutes": round(elapsed_minutes, 2),
        "peak_allocated_vram_gb": round(peak_vram, 2),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }

    with open(
        ADAPTER_DIR / "training_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, default=str)

    print(f"\nAdapter saved at: {ADAPTER_DIR}")
    print(f"Metrics saved at: {ADAPTER_DIR / 'training_metrics.json'}")
    print("\nFine-tuning completed!")


if __name__ == "__main__":
    main()