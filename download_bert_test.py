from transformers import AutoTokenizer, AutoModel

# Hugging Face model repository
model_name = "google-bert/bert-base-uncased"

# Download pretrained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download pretrained BERT model and weights
model = AutoModel.from_pretrained(model_name)

# Save downloaded model locally
model.save_pretrained("./bert_pretrained")
tokenizer.save_pretrained("./bert_pretrained")

print("Successfully downloaded pretrained BERT!")

# Display model information
print("Model type:", model.config.model_type)
print("Number of layers:", model.config.num_hidden_layers)
print("Hidden dimensions:", model.config.hidden_size)
print("Vocabulary size:", model.config.vocab_size)

# Count model parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Total parameters:", total_parameters)
