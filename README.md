
# NLP Final Project: BERT Sentiment Classification

## Prerequisites

**A Groq API key is required for the LLM-powered components of this project. You will need to create a Groq API key and configure it in your `.env` file using the template provided in `.env.example`.  A GPU-powered computer or an equivalent environment (such as Google Colab with GPU runtime enabled) is highly recommended for this project.**

GPU acceleration is important because fine-tuning BERT involves computationally intensive matrix operations across millions of model parameters. Using a GPU significantly reduces training time compared to running the training process on a CPU.

## 1. Environment Setup

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

**Git Bash:**

```bash
source venv\Scripts\activate
```

**Linux:**

```bash
source venv/bin/activate
```


## 2. Install Dependencies

Install all required packages and libraries:

```bash
pip install -r requirements.txt
```

## 3. Configure Groq API Key

This project uses Groq's API for its LLM-powered components. Follow these steps to configure your API key:

1. Create an account or sign in to [Groq Console](https://console.groq.com/).
2. Navigate to the API Keys section and generate a new API key.
3. Open the `.env.example` file in the project root directory and follow the provided template to configure your `GROQ_API_KEY` and `GROQ_MODEL`.  Save the configured file as `.env` in the project root directory.
4. Verify your Groq API configuration by running:

   ```bash
   python test_groq.py

If successful, the script will print a confirmation message, the configured model name, and a generated response.

**Note:** Keep your API key private. Do not commit or upload your `.env` file to GitHub or other public repositories.

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

## 5. Generate Challenge Dataset

Run the following script to generate a challenge dataset containing linguistically perturbed versions of the original SST-2 sentences:

```bash
python create_challenge_dataset.py
```

Once the script finishes executing, the generated dataset will be saved as `challenge_dataset.jsonl` in the project directory. This dataset will be used in subsequent experiments to evaluate the model's robustness against linguistic perturbations.

## 6. Evaluate Model Robustness Against Linguistic Perturbations

After generating the challenge dataset, run the following script to evaluate the fine-tuned BERT model against the linguistically perturbed sentences:

```bash
python evaluate_challenge_dataset.py --model_path fine_tuned_bert/ --dataset_path challenge_dataset.jsonl
```


## 6. Project Workflow

The overall workflow is:

1. Create and activate the Python virtual environment.
2. Install the required dependencies.
3. Create and configure the `.env` file with your Groq API key and model settings, then run `test_groq.py` to verify the API connection.
4. Download the pretrained BERT model.
5. Fine-tune BERT on the SST-2 sentiment classification dataset.
6. Evaluate and save the fine-tuned model on the original SST-2 validation dataset (clean, unmodified sentences) and save the model.  This establishes baseline performance, which serves as a reference point for comparing how different linguistic perturbations affect the model's sentiment classification accuracy and overall robustness.
7. Generate the challenge dataset containing linguistically perturbed versions of the original SST-2 sentences for subsequent robustness evaluation.
8. Evaluate the fine-tuned BERT model on the generated challenge dataset to assess its robustness against linguistic perturbations.