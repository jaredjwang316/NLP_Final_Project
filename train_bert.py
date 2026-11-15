"""Fine-tune BERT on SST-2 with full fine-tuning, LoRA, or QLoRA.

Colab usage:
    pip install peft bitsandbytes
    python train_bert.py --method qlora --drive_dir /content/drive/MyDrive/nlp_outputs
QLoRA requires an NVIDIA GPU; full and lora run on CPU or GPU.
"""

import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

DEFAULT_METHODS = {
    "full": {"lr": 2e-5, "batch_size": 16},
    "lora": {"lr": 1e-4, "batch_size": 32},
    "qlora": {"lr": 2e-4, "batch_size": 32},
}

DEFAULT_OUTPUT_DIRS = {
    "full": "./fine_tuned_bert",
    "lora": "./fine_tuned_bert_lora",
    "qlora": "./fine_tuned_bert_qlora",
}


# Arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["full", "lora", "qlora"], default="full")
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--drive_dir", default=None)
    return parser.parse_args()


# Model loading
def load_model(method, model_path, num_labels=2):
    """Load the model, applying 4-bit quantization for qlora."""
    quantization_config = None
    if method == "qlora":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        quantization_config=quantization_config,
    )


# LoRA setup
def build_lora_model(model, method, r, alpha, dropout):
    """Wrap the model with LoRA adapters (no-op for full fine-tuning)."""
    if method == "full":
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    if method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# Metrics
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 for the positive class."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    accuracy = np.mean(predictions == labels)
    true_positive = np.sum((predictions == 1) & (labels == 1))
    false_positive = np.sum((predictions == 1) & (labels == 0))
    false_negative = np.sum((predictions == 0) & (labels == 1))
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Training arguments
def build_training_args(method, args):
    """Build TrainingArguments for the chosen method."""
    return TrainingArguments(
        output_dir=args.output_dir or f"./bert_training_results_{method}",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )


# Main
def main():
    """Run the fine-tuning pipeline for the chosen method."""
    args = parse_args()
    defaults = DEFAULT_METHODS[args.method]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    output_path = args.output_dir or DEFAULT_OUTPUT_DIRS[args.method]

    print(f"Method: {args.method} | Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.method, args.model)
    model = build_lora_model(model, args.method, args.lora_r, args.lora_alpha, args.lora_dropout)

    print("Loading and tokenizing SST-2...")
    dataset = load_dataset("stanfordnlp/sst2")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = build_training_args(args.method, args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating on clean SST-2 validation data...")
    results = trainer.evaluate()
    for metric, value in results.items():
        print(metric, ":", value)

    print(f"Saving to {output_path}...")
    if args.method == "full":
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
    elif args.method == "lora":
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{output_path}_merged")
        tokenizer.save_pretrained(f"{output_path}_merged")
    else:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    if args.drive_dir:
        import shutil

        shutil.copytree(output_path, args.drive_dir, dirs_exist_ok=True)
        print(f"Copied outputs to {args.drive_dir}")

    print("Training script completed successfully!")


if __name__ == "__main__":
    main()
