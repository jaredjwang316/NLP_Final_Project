
# NLP Final Project: BERT Sentiment Classification

## Prerequisites

**A GPU-powered computer or an equivalent environment (such as Google Colab with GPU runtime enabled) is highly recommended for this project.**

GPU acceleration is important because fine-tuning BERT involves computationally intensive matrix operations across millions of model parameters. Using a GPU significantly reduces training time compared to running the training process on a CPU.

## 1. Environment Setup

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment using Git Bash:

```bash
source venv/Scripts/activate
```

## 2. Install Dependencies

Install all required packages and libraries:

```bash
pip install -r requirements.txt
```

## 3. Download the Pretrained BERT Model

Run the following script to download the pretrained `bert-base-uncased` model and its tokenizer from Hugging Face:

```bash
python download_bert_test.py
```

The pretrained model and tokenizer will be saved locally in a folder named `bert_pretrained/`.

## 4. Fine-Tune BERT for Sentiment Classification

Run the training script:

```bash
python train_bert.py
```

This script fine-tunes the pretrained BERT model on the SST-2 dataset for binary sentiment classification (Negative or Positive). It also evaluates the model using accuracy, precision, recall, and F1-score on the validation dataset.

After training is completed, the fine-tuned model and tokenizer will be saved in:

```text
fine_tuned_bert/
```

## 5. Project Workflow

The overall workflow is:

1. Create and activate the Python virtual environment.
2. Install the required dependencies.
3. Download the pretrained BERT model.
4. Fine-tune BERT on the SST-2 sentiment classification dataset.
5. Evaluate and save the fine-tuned model on the original SST-2 validation dataset (clean, unmodified sentences) and save the model.

**Note:** The `bert_pretrained/` folder must be generated before running `train_bert.py`, since the training script loads the pretrained model from that directory.