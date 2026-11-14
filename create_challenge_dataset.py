import random
from datasets import load_dataset

# --------------------------------------------------
# 1. Load SST-2 validation dataset
# --------------------------------------------------
dataset = load_dataset("stanfordnlp/sst2")
validation_data = dataset["validation"]
print("Total validation sentences:", len(validation_data))

# Make typo generation reproducible
random.seed(42)

# --------------------------------------------------
# 2. Introduce a typo into an individual word
# --------------------------------------------------
def introduce_typo(word):

    if len(word) < 4:
        return word
    # Randomly select one of four typo types
    typo_type = random.randint(1, 4)

    # Select a character position
    position = random.randint(1, len(word) - 2)

    # Type 1: Character deletion
    if typo_type == 1:
        modified_word = word[:position] + word[position + 1:]
    # Type 2: Character insertion
    elif typo_type == 2:
        letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        modified_word = word[:position] + letter + word[position:]
    # Type 3: Character substitution
    elif typo_type == 3:
        letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        while letter == word[position].lower():
            letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        modified_word = word[:position] + letter + word[position + 1:]
    # Type 4: Character transposition
    else:
        modified_word = (
            word[:position]
            + word[position + 1]    #transposition occurs here
            + word[position]
            + word[position + 2:]
        )

    return modified_word


# --------------------------------------------------
# 3. Introduce a typo into one word in a sentence
# --------------------------------------------------
def introduce_sentence_typo(sentence):

    words = sentence.split()
    # Find words that can be modified
    candidates = []
    for i in range(len(words)):
        clean_word = words[i].strip(".,!?;:\"'()")
        if len(clean_word) >= 4:
            candidates.append(i)

    # Return original sentence if no suitable word exists
    if len(candidates) == 0:
        return sentence

    # Randomly select one word
    selected_index = random.choice(candidates)
    original_word = words[selected_index]

    # Separate punctuation from the word
    punctuation_prefix = ""
    punctuation_suffix = ""

    while len(original_word) > 0 and original_word[0] in ".,!?;:\"'()":
        punctuation_prefix += original_word[0]
        original_word = original_word[1:]

    while len(original_word) > 0 and original_word[-1] in ".,!?;:\"'()":
        punctuation_suffix = original_word[-1] + punctuation_suffix
        original_word = original_word[:-1]

    # Generate modified word
    modified_word = introduce_typo(original_word)

    # Replace the original word
    words[selected_index] = punctuation_prefix + modified_word + punctuation_suffix

    return " ".join(words)


# --------------------------------------------------
# 4. Create typo challenge dataset
# --------------------------------------------------
challenge_dataset = []

for i in range(len(validation_data)):
    # Original SST-2 sentence and gold label
    original_sentence = validation_data[i]["sentence"]
    original_label = validation_data[i]["label"]

    # Generate typo-modified sentence
    modified_sentence = introduce_sentence_typo(original_sentence)

    # Store original and modified examples together
    challenge_dataset.append({
        "original_sentence": original_sentence,
        "modified_sentence": modified_sentence,
        "original_label": original_label,
        "modified_label": original_label,
        "perturbation_type": "typo"
    })

    print("\n----------------------------------------")
    print("Sentence index:", i)
    print("Original sentence:", original_sentence)
    print("Modified sentence:", modified_sentence)
    print("Original gold label:", original_label)
    print("Modified gold label:", original_label)


# --------------------------------------------------
# 5. Display challenge dataset information
# --------------------------------------------------
print("\n========================================")
print("TYPO CHALLENGE DATASET CREATED")
print("Total original examples:", len(challenge_dataset))
print("Total modified examples:", len(challenge_dataset))
print("Total paired examples:", len(challenge_dataset))
print("========================================")
