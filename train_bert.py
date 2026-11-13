import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# --------------------------------------------------
# 1. Load the pretrained BERT model and tokenizer
# --------------------------------------------------

# This assumes your downloaded model is saved in
# a folder named "bert_pretrained" in your project directory.
MODEL_NAME = "./bert_pretrained"

print("Loading pretrained BERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add a classification head with 2 output labels:
# 0 = Negative, 1 = Positive
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

print("BERT model loaded!")


# --------------------------------------------------
# 2. Load the SST-2 dataset
# --------------------------------------------------

print("Loading SST-2 dataset...")

dataset = load_dataset("stanfordnlp/sst2")

train_data = dataset["train"]
validation_data = dataset["validation"]

print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))


# --------------------------------------------------
# 3. Tokenize the sentences
# --------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=128
    )

print("Tokenizing training data...")

tokenized_train = train_data.map(
    tokenize_function,
    batched=True
)

print("Tokenizing validation data...")

tokenized_validation = validation_data.map(
    tokenize_function,
    batched=True
)

# Trainer expects the target column to be named "labels".
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_validation = tokenized_validation.rename_column("label", "labels")

# Remove columns that the model does not need.
tokenized_train = tokenized_train.remove_columns(["sentence", "idx"])
tokenized_validation = tokenized_validation.remove_columns(["sentence", "idx"])

# Pad sentences dynamically when they are grouped into batches.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Tokenization completed!")


# --------------------------------------------------
# 4. Define evaluation metrics
# --------------------------------------------------

def compute_metrics(eval_pred):

    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert the two output scores into a predicted label.
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy.
    accuracy = np.mean(predictions == labels)

    # Calculate precision, recall, and F1 for positive class (label 1).
    true_positive = np.sum((predictions == 1) & (labels == 1))
    false_positive = np.sum((predictions == 1) & (labels == 0))
    false_negative = np.sum((predictions == 0) & (labels == 1))

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# 5. Configure the fine-tuning process
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./bert_training_results",

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    num_train_epochs=3,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    logging_steps=100,
    save_total_limit=2,

    seed=42,
    report_to="none"
)


# --------------------------------------------------
# 6. Create the Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# --------------------------------------------------
# 7. Fine-tune BERT
# --------------------------------------------------

print("\nStarting BERT fine-tuning...")

trainer.train()

print("\nFine-tuning completed!")


# --------------------------------------------------
# 8. Evaluate the fine-tuned model
# --------------------------------------------------

print("\nEvaluating on clean SST-2 validation data...")

results = trainer.evaluate()

print("\nValidation results:")

for metric, value in results.items():
    print(metric, ":", value)


# --------------------------------------------------
# 9. Save the fine-tuned model
# --------------------------------------------------

output_path = "./fine_tuned_bert"

trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)

print("\nFine-tuned model saved to:", output_path)
print("Training script completed successfully!")
