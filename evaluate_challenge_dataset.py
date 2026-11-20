
"""
Evaluate a fine-tuned BERT sentiment classifier on a
perturbed SST-2 challenge dataset.

Supports:
    - Full fine-tuning
    - LoRA adapters
    - QLoRA adapters

Example:
    python evaluate_challenge_dataset.py \
        --model_path ./fine_tuned_bert \
        --dataset_path ./challenge_dataset.jsonl
"""

import argparse
import json
import os

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="./fine_tuned_bert"
    )

    parser.add_argument(
        "--dataset_path",
        default="./challenge_dataset.jsonl"
    )

    parser.add_argument(
        "--output",
        default="./challenge_predictions.jsonl"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    """Load a full BERT model or a PEFT adapter."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Check whether the directory contains a PEFT adapter.
    adapter_config = os.path.join(
        model_path,
        "adapter_config.json"
    )

    if os.path.isfile(adapter_config):

        from peft import AutoPeftModelForSequenceClassification

        print("Loading PEFT adapter...")

        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path
        )

    else:

        print("Loading full fine-tuned model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

    return model, tokenizer


def load_challenge_dataset(dataset_path):
    """Read the JSONL challenge dataset."""

    rows = []

    with open(dataset_path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                rows.append(json.loads(line))

    return rows


def predict_sentiment(
    sentences,
    model,
    tokenizer,
    device,
    batch_size,
    max_length
):
    """Generate sentiment predictions using BERT."""

    predictions = []

    model.eval()

    for start in range(0, len(sentences), batch_size):

        batch = sentences[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

        predictions.extend(
            batch_predictions.cpu().tolist()
        )

    return predictions


def calculate_metrics(true_labels, predictions):
    """Calculate accuracy, precision, recall, and F1."""

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(
            true_labels,
            predictions
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def print_metrics(title, labels, predictions):

    metrics = calculate_metrics(labels, predictions)

    print(f"\n========== {title} ==========")

    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")

    print("\nClassification report:")

    print(classification_report(
        labels,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    print("Confusion matrix:")

    print(confusion_matrix(
        labels,
        predictions
    ))

    return metrics


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model path:", args.model_path)
    print("Dataset path:", args.dataset_path)

    # Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(
        args.model_path
    )

    model.to(device)

    # Load challenge dataset.
    rows = load_challenge_dataset(
        args.dataset_path
    )

    print("\nTotal challenge examples:", len(rows))

    # Extract original and modified sentences.
    original_sentences = [
        row["original_sentence"] for row in rows
    ]

    modified_sentences = [
        row["modified_sentence"] for row in rows
    ]

    original_labels = [
        row["original_label"] for row in rows
    ]

    modified_labels = [
        row["modified_label"] for row in rows
    ]

    # Generate predictions.
    print("\nEvaluating original sentences...")

    original_predictions = predict_sentiment(
        original_sentences,
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_length
    )

    print("Evaluating modified sentences...")

    modified_predictions = predict_sentiment(
        modified_sentences,
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_length
    )

    # Overall evaluation.
    original_metrics = print_metrics(
        "CLEAN ORIGINAL SENTENCES",
        original_labels,
        original_predictions
    )

    modified_metrics = print_metrics(
        "PERTURBED CHALLENGE SENTENCES",
        modified_labels,
        modified_predictions
    )

    # Compare performance.
    print("\n========== PERFORMANCE COMPARISON ==========")

    for metric in original_metrics:

        clean = original_metrics[metric]
        perturbed = modified_metrics[metric]

        difference = perturbed - clean

        print(
            f"{metric.capitalize()}: "
            f"Clean = {clean:.4f}, "
            f"Perturbed = {perturbed:.4f}, "
            f"Difference = {difference:+.4f}"
        )

    # Evaluate examples containing each transformation.
    print("\n========== RESULTS BY TRANSFORMATION ==========")

    transform_names = sorted({
        transform
        for row in rows
        for transform in row["transforms"]
    })

    for transform in transform_names:

        indices = [
            i for i, row in enumerate(rows)
            if transform in row["transforms"]
        ]

        labels = [
            modified_labels[i] for i in indices
        ]

        predictions = [
            modified_predictions[i] for i in indices
        ]

        metrics = calculate_metrics(
            labels,
            predictions
        )

        print(f"\nTransformation: {transform}")
        print("Examples:", len(indices))
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")

    # Save individual predictions.
    print("\nSaving individual predictions...")

    with open(args.output, "w", encoding="utf-8") as f:

        for i, row in enumerate(rows):

            result = {
                "id": row["id"],
                "original_sentence": row["original_sentence"],
                "modified_sentence": row["modified_sentence"],
                "original_label": row["original_label"],
                "modified_label": row["modified_label"],
                "original_prediction": original_predictions[i],
                "modified_prediction": modified_predictions[i],
                "transforms": row["transforms"],
                "intensity": row["intensity"],
                "original_correct": (
                    original_predictions[i] == original_labels[i]
                ),
                "modified_correct": (
                    modified_predictions[i] == modified_labels[i]
                )
            }

            f.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print("\nPredictions saved to:", args.output)
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
