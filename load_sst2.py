from datasets import load_dataset

print("Downloading/loading SST-2...")

dataset = load_dataset("stanfordnlp/sst2")

print("\nDataset successfully loaded!")

# Access the training split
train_data = dataset["train"]

print("\nNumber of training examples:", len(train_data))

print("\nFirst five training examples:\n")
for i in range(5):

    example = train_data[i]

    sentence = example["sentence"]

    label = example["label"]

    print("Example:", i)
    print("Sentence:", sentence)
    print("Gold label:", label)

    if label == 0:
        print("Sentiment: Negative")
    elif label == 1:
        print("Sentiment: Positive")

    print("----------------------------")


# Verify validation split
print("\nValidation examples:", len(dataset["validation"]))

# Verify test split
print("Test examples:", len(dataset["test"]))

print("\nSST-2 loading test completed!")