"""Build a challenge dataset from SST-2 validation with randomized perturbations.

Each sentence gets k distinct transforms (k in [k_min, k_max]); each transform
draws its own intensity p per sentence, and every candidate position is
perturbed independently with probability p. Deterministic for a given seed.

Semantic rewording and negation are generated using Groq.  These LLM 
transformations do not use an intensity probability.

Deterministic for traditional transformations given a seed. LLM outputs may 
vary between runs.
"""

import argparse
import json
import random
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

load_dotenv()   #Load the environment variables from the .env file

PUNCTUATION = ".,!?;:\"'()"

DIACRITIC_MAP = {
    "a": "àáâä", "e": "éèêë", "i": "ïíîì", "o": "öóôò",
    "u": "üúûù", "n": "ñ", "c": "ç", "y": "ÿ",
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./challenge_dataset.jsonl")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_samples", type=int, default=0)
    return parser.parse_args()


# Character-level transforms
def typo_word(word, rng):
    """Apply one random character-level typo to a word."""
    if len(word) < 4:
        return word
    position = rng.randint(1, len(word) - 2)
    op = rng.choice(["delete", "insert", "substitute", "transpose"])
    if op == "delete":
        return word[:position] + word[position + 1:]
    if op == "insert":
        letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:position] + letter + word[position:]
    if op == "substitute":
        letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
        while letter == word[position].lower():
            letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:position] + letter + word[position + 1:]
    return word[:position] + word[position + 1] + word[position] + word[position + 2:]


def strip_punctuation(word):
    """Return the leading and trailing punctuation and the core word."""
    prefix = ""
    while word and word[0] in PUNCTUATION:
        prefix += word[0]
        word = word[1:]
    suffix = ""
    while word and word[-1] in PUNCTUATION:
        suffix = word[-1] + suffix
        word = word[:-1]
    return prefix, word, suffix


def transform_typo(sentence, p, rng):
    """Apply character-level typos to each candidate word with probability p."""
    words = sentence.split()
    for i, word in enumerate(words):
        if rng.random() < p:
            prefix, core, suffix = strip_punctuation(word)
            words[i] = prefix + typo_word(core, rng) + suffix
    return " ".join(words)


def transform_diacritic(sentence, p, rng):
    """Replace plain letters with accented lookalikes with probability p."""
    words = sentence.split()
    for i, word in enumerate(words):
        prefix, core, suffix = strip_punctuation(word)
        if len(core) < 3:
            continue
        chars = list(core)
        for j, ch in enumerate(chars):
            if ch.lower() in DIACRITIC_MAP and rng.random() < p:
                chars[j] = rng.choice(DIACRITIC_MAP[ch.lower()])
        words[i] = prefix + "".join(chars) + suffix
    return " ".join(words)


def transform_whitespace(sentence, p, rng):
    """Collapse, double, or tab-replace spaces with probability p."""
    parts = sentence.split(" ")
    out = []
    for part in parts:
        if part and rng.random() < p:
            op = rng.choice(["collapse", "double", "tab"])
            if op == "collapse":
                out.append(part.strip())
                continue
            if op == "double":
                out.append(part + "  ")
                continue
            out.append(part + "\t")
            continue
        out.append(part)
    return " ".join(out)


def transform_punctuation(sentence, p, rng):
    """Add, swap, or remove punctuation with probability p."""
    words = sentence.split()
    for i, word in enumerate(words):
        if rng.random() >= p:
            continue
        prefix, core, suffix = strip_punctuation(word)
        op = rng.choice(["add", "swap", "remove"])
        if op == "add":
            suffix = rng.choice([",", ".", ";", "!", "--", "..."])
        elif op == "swap" and suffix:
            suffix = rng.choice([",", ".", ";", "!", "?"])
        elif op == "remove":
            suffix = ""
        words[i] = prefix + core + suffix
    return " ".join(words)

# LLM-based semantic transformations
SEMANTIC_SYSTEM_PROMPT = """
You are an expert NLP data augmentation assistant.

You are generating challenging examples for an SST-2 sentiment
classification robustness experiment.

SST-2 is a movie review sentiment dataset with two labels:
0 = negative sentiment
1 = positive sentiment

You will receive an original movie review and a requested transformation.

Your job is to generate a natural, grammatically correct sentence
that follows the requested transformation.

General requirements:
- Preserve the original topic and relevant information.
- Use natural English.
- Do not introduce unrelated information.
- Avoid awkward or unnatural expressions.
- Return only the requested structured JSON output.
"""

def generate_llm_transformation(sentence, label, transformation, client):
    """Generate a semantic transformation using Groq."""
    instruction = ""
    if transformation == "semantic_rewording":

        instruction = """
        Generate a semantic rewording of the original movie review.

        Requirements:
        1. Preserve the original meaning and sentiment.
        2. Use different vocabulary, synonyms, and sentence structures.
        3. Do not simply replace one word with another.
        4. Preserve the important information and opinions.
        5. Make the result sound like a naturally written movie review.
        6. Do not reverse or change the sentiment.
        """
    elif transformation == "negation":

        opposite_label = 1 - label

        instruction = f"""
        Generate a negation-based version of the original movie review
        that reverses its overall sentiment.

        Original sentiment label: {label}
        Required sentiment label: {opposite_label}

        Requirements:
        1. Use natural negation or negative constructions.
        2. Reverse the overall sentiment of the original review.
        3. Preserve the original topic and as much of its content as reasonably possible.
        4. Do not simply insert the word "not" without changing the actual sentiment.
        5. Make the result sound like a naturally written movie review.
        6. If a meaningful sentiment reversal cannot be generated,
           return the original sentence and mark success as false.
        """
    else:
        raise ValueError("Unknown transformation: " + transformation)

    try:
        # Call Groq's Chat Completions API (an API endpoint for sending prompts to an LLM and receiving its 
        # generated response) and store the returned response object in the response variable.
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),  #specifiy the LLM that Groq should use to generate the response
            messages=[
                {
                    "role": "system",   # System message that establishes the LLM's role/behavior
                    "content": SEMANTIC_SYSTEM_PROMPT
                },
                {
                    "role": "user", # User message that provides the actual/concrete task for the LLM to do
                    "content": (
                        instruction + "\n\nOriginal sentence: " + sentence
                    )
                }
            ],
            response_format={
                # Tell LLM to return its response in a specific JSON structure instead of an ordinary conversational text
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_transformation",
                    "strict": True, # model must strictly adhere to the JSON format defined below
                    "schema": {
                        "type": "object",
                        "properties": {
                            "transformed_sentence": {
                                "type": "string"
                            },
                            "success": {
                                "type": "boolean"
                            },
                            "sentiment_label": {
                                "type": "integer",
                                "enum": [0, 1]  #sentiment label is an integer and can be values 0 or 1
                            }
                        },
                        "required": [
                            "transformed_sentence",
                            "success",
                            "sentiment_label"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            reasoning_effort = "low",
            max_completion_tokens=1500   # maximum number of tokens the model can generate per response
        )

        # Pause for 10 seconds after a successful API request.
        time.sleep(10)

        # Convert the LLM's response (string) to a Python dictionary
        result = json.loads(
            response.choices[0].message.content
        )

        transformed = result["transformed_sentence"].strip()
        success = result["success"]
        generated_label = result["sentiment_label"]

        if not success or not transformed:
            return sentence

        # Validate the expected sentiment label.
        expected_label = label
        if transformation == "negation":
            expected_label = 1 - label

        if generated_label != expected_label:
            return sentence

        return transformed

    except Exception as e:
        print(f"Groq transformation failed: {e}")
        return sentence


def transform_word_substitute(sentence, label, client):
    """Generate a semantic rewording while preserving sentiment."""
    return generate_llm_transformation(
        sentence,
        label,
        "semantic_rewording",
        client
    )


def transform_ngram_substitute(sentence, label, client):
    """Generate a negation-based transformation."""
    return generate_llm_transformation(
        sentence,
        label,
        "negation",
        client
    )


def transform_token_swap(sentence, p, rng):
    """Swap adjacent word pairs with probability p, skipping the first word."""
    words = sentence.split()
    i = 1
    while i < len(words) - 1:
        if rng.random() < p:
            words[i], words[i + 1] = words[i + 1], words[i]
            i += 2
        else:
            i += 1
    return " ".join(words)


def transform_token_delete(sentence, p, rng):
    """Delete words with probability p while keeping at least one word."""
    words = sentence.split()
    keep = [w for w in words if rng.random() >= p]
    if len(keep) < max(1, len(words) // 2):
        return " ".join(words)
    return " ".join(keep)


TRANSFORMS = [
    ("whitespace", transform_whitespace),
    ("punctuation", transform_punctuation),
    ("typo", transform_typo),
    ("diacritic", transform_diacritic),
    ("word_substitute", transform_word_substitute),
    ("token_swap", transform_token_swap),
    ("token_delete", transform_token_delete),
    ("ngram_substitute", transform_ngram_substitute),
]

# Pipeline
def augment_sentence(sentence, label, rng, client, k_min=1, k_max=3):
    """Apply k randomly chosen transforms, each with its own random intensity."""
    k = rng.randint(k_min, k_max)
    chosen = rng.sample(TRANSFORMS, k)

    applied = []
    intensities = {}

    for name, fn in chosen:
        if name == "word_substitute":
            modified = fn(sentence, label, client)
        elif name == "ngram_substitute":
            modified = fn(sentence, label, client)
            if modified != sentence:    #negation reverses the sentiment label
                label = 1 - label
        else:
            p = rng.uniform(0.1, 0.6)
            modified = fn(sentence, p, rng)

            if modified != sentence:
                intensities[name] = round(p, 2)
        
        if modified != sentence:
            applied.append(name)
            sentence = modified
    return sentence, label, applied, intensities


def main():
    """Generate the challenge dataset and write it to JSONL."""
    args = parse_args()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")

    groq_model = os.getenv("GROQ_MODEL")
    if not groq_model:
        raise ValueError("GROQ_MODEL not found. Check your .env file.")

    # Create a groq client object
    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    rng = random.Random(args.seed)
    dataset = load_dataset("stanfordnlp/sst2")[args.split]
    print(f"Total {args.split} sentences:", len(dataset))

    rows = []
    for i in range(len(dataset)):
        original = dataset[i]["sentence"]
        label = dataset[i]["label"]
        modified, modified_label, transforms, intensities = augment_sentence(original, label, rng, client, args.k_min, args.k_max)
        rows.append({
            "id": i,
            "original_sentence": original,
            "modified_sentence": modified,
            "original_label": label,
            "modified_label": modified_label,
            "transforms": transforms,
            "intensity": intensities,
            "seed": args.seed,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Challenge dataset written to:", args.output)
    print("Total rows:", len(rows))
    transform_counts = {}
    for row in rows:
        for t in row["transforms"]:
            transform_counts[t] = transform_counts.get(t, 0) + 1
    print("Transform counts:", transform_counts)

    for i in range(min(args.print_samples, len(rows))):
        row = rows[i]
        print("\n---")
        print("Original:", row["original_sentence"])
        print("Modified:", row["modified_sentence"])
        print("Transforms:", row["transforms"], row["intensity"])


if __name__ == "__main__":
    main()
