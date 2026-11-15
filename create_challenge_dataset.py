"""Build a challenge dataset from SST-2 validation with randomized perturbations.

Each sentence gets k distinct transforms (k in [k_min, k_max]); each transform
draws its own intensity p per sentence, and every candidate position is
perturbed independently with probability p. Deterministic for a given seed.
"""

import argparse
import json
import random
from datasets import load_dataset

PUNCTUATION = ".,!?;:\"'()"

DIACRITIC_MAP = {
    "a": "àáâä", "e": "éèêë", "i": "ïíîì", "o": "öóôò",
    "u": "üúûù", "n": "ñ", "c": "ç", "y": "ÿ",
}

SYNONYMS = {
    "good": ["nice", "fine", "pleasant"], "bad": ["poor", "awful", "terrible"],
    "great": ["wonderful", "excellent", "superb"], "movie": ["film", "picture"],
    "film": ["movie", "picture"], "boring": ["dull", "tedious"],
    "funny": ["amusing", "hilarious"], "beautiful": ["gorgeous", "stunning"],
    "terrible": ["awful", "horrid"], "fun": ["enjoyable", "amusing"],
    "smart": ["clever", "intelligent"], "acting": ["performance", "portrayal"],
    "story": ["plot", "tale"], "worst": ["poorest", "lousiest"],
    "best": ["finest", "greatest"], "actor": ["performer"],
    "performance": ["acting", "turn"], "surprising": ["startling", "unexpected"],
    "charming": ["delightful", "enchanting"], "clever": ["smart", "witty"],
    "dull": ["boring", "lifeless"], "amazing": ["astonishing", "incredible"],
    "wonderful": ["marvelous", "delightful"], "horrible": ["dreadful", "awful"],
}

NGRAM_MAP = {
    ("not", "good"): ["bad"], ("very", "good"): ["great"], ("not", "bad"): ["decent"],
    ("a", "lot", "of"): ["plenty", "of"], ("the", "film"): ["this", "movie"],
    ("the", "movie"): ["the", "film"], ("this", "movie"): ["this", "film"],
    ("one", "of", "the"): ["among", "the"], ("worth", "watching"): ["worth", "seeing"],
    ("worth", "seeing"): ["worth", "watching"], ("so", "bad"): ["terrible"],
    ("as", "well", "as"): ["along", "with"], ("in", "order", "to"): ["to"],
    ("kind", "of"): ["sort", "of"], ("it", "is"): ["it's"], ("it's", "a"): ["it", "is"],
    ("there", "is"): ["there's"], ("do", "not"): ["don't"], ("does", "not"): ["doesn't"],
    ("can", "not"): ["can't"], ("will", "not"): ["won't"],
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


# Word-level transforms
def transform_word_substitute(sentence, p, rng):
    """Replace words from the synonym map with probability p."""
    words = sentence.split()
    for i, word in enumerate(words):
        if rng.random() >= p:
            continue
        prefix, core, suffix = strip_punctuation(word)
        lowered = core.lower()
        if lowered in SYNONYMS:
            replacement = rng.choice(SYNONYMS[lowered])
            if core[0].isupper():
                replacement = replacement.capitalize()
            words[i] = prefix + replacement + suffix
    return " ".join(words)


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


# N-gram transforms
def transform_ngram_substitute(sentence, p, rng):
    """Replace 2-3 word n-grams from the paraphrase map with probability p."""
    words = sentence.split()
    i = 0
    out = []
    while i < len(words):
        matched = False
        for n in (3, 2):
            if i + n <= len(words):
                gram = tuple(w.lower().strip(PUNCTUATION) for w in words[i:i + n])
                if gram in NGRAM_MAP and rng.random() < p:
                    replacement = rng.choice(NGRAM_MAP[gram])
                    out.extend(replacement)
                    i += n
                    matched = True
                    break
        if not matched:
            out.append(words[i])
            i += 1
    return " ".join(out)


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
def augment_sentence(sentence, rng, k_min=1, k_max=3):
    """Apply k randomly chosen transforms, each with its own random intensity."""
    k = rng.randint(k_min, k_max)
    chosen = rng.sample(TRANSFORMS, k)
    applied = []
    intensities = {}
    for name, fn in chosen:
        p = rng.uniform(0.1, 0.6)
        modified = fn(sentence, p, rng)
        if modified != sentence:
            applied.append(name)
            intensities[name] = round(p, 2)
            sentence = modified
    return sentence, applied, intensities


def main():
    """Generate the challenge dataset and write it to JSONL."""
    args = parse_args()
    rng = random.Random(args.seed)
    dataset = load_dataset("stanfordnlp/sst2")[args.split]
    print(f"Total {args.split} sentences:", len(dataset))

    rows = []
    for i in range(len(dataset)):
        original = dataset[i]["sentence"]
        label = dataset[i]["label"]
        modified, transforms, intensities = augment_sentence(original, rng, args.k_min, args.k_max)
        rows.append({
            "id": i,
            "original_sentence": original,
            "modified_sentence": modified,
            "original_label": label,
            "modified_label": label,
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
