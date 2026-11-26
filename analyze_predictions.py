"""Analyze saved challenge predictions by intensity and error direction.

Modes:
    intensity  - accuracy binned by number of transforms and mean intensity
    errors     - error-direction analysis per transform with examples

Example:
    python analyze_predictions.py --mode intensity --predictions ./challenge_predictions.jsonl
"""

import argparse
import json

from collections import defaultdict


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="intensity", choices=["intensity", "errors"])
    parser.add_argument("--predictions", default="./challenge_predictions.jsonl")
    parser.add_argument("--output", default="./analysis_results.jsonl")
    return parser.parse_args()


def load_rows(path):
    """Load prediction rows from a JSONL file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean_intensity(row):
    """Return the mean intensity across the row's deterministic transforms."""
    values = list(row["intensity"].values())
    if not values:
        return None
    return sum(values) / len(values)


def bucket_label(value):
    """Map a numeric value to a labeled bucket."""
    if value is None:
        return "none"
    return f"{value:.1f}-{value + 0.1:.1f}"


def run_intensity(rows):
    """Report accuracy by transform count and by mean intensity bucket."""
    print("========== ACCURACY BY NUMBER OF TRANSFORMS ==========")

    by_count = defaultdict(lambda: [0, 0])
    for row in rows:
        count = len(row["transforms"])
        by_count[count][0] += 1
        if row["modified_correct"]:
            by_count[count][1] += 1

    for count in sorted(by_count):
        total, correct = by_count[count]
        print(f"Transforms: {count}")
        print("Examples:", total)
        print(f"Accuracy: {correct / total:.4f}")

    print("\n========== ACCURACY BY MEAN INTENSITY ==========")

    by_bucket = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = bucket_label(mean_intensity(row))
        by_bucket[bucket][0] += 1
        if row["modified_correct"]:
            by_bucket[bucket][1] += 1

    order = ["none", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6"]

    for bucket in order:
        if bucket not in by_bucket:
            continue
        total, correct = by_bucket[bucket]
        print(f"Intensity: {bucket}")
        print("Examples:", total)
        print(f"Accuracy: {correct / total:.4f}")


def run_errors(rows):
    """Report error direction per transform with example sentences."""
    print("========== ERROR DIRECTION BY TRANSFORMATION ==========")

    by_transform = defaultdict(lambda: defaultdict(list))

    for row in rows:
        for transform in row["transforms"]:
            if row["original_correct"] and not row["modified_correct"]:
                by_transform[transform]["clean_right_perturbed_wrong"].append(row)
            elif not row["original_correct"] and row["modified_correct"]:
                by_transform[transform]["clean_wrong_perturbed_right"].append(row)
            elif not row["original_correct"] and not row["modified_correct"]:
                by_transform[transform]["wrong_both"].append(row)
            else:
                by_transform[transform]["correct_both"].append(row)

    for transform in sorted(by_transform):
        groups = by_transform[transform]
        total = sum(len(g) for g in groups.values())
        degraded = len(groups["clean_right_perturbed_wrong"])
        recovered = len(groups["clean_wrong_perturbed_right"])
        print(f"\nTransformation: {transform}")
        print("Examples:", total)
        print(f"Correct both: {len(groups['correct_both'])}")
        print(f"Clean right, perturbed wrong: {degraded}")
        print(f"Clean wrong, perturbed right: {recovered}")
        print(f"Wrong both: {len(groups['wrong_both'])}")
        print(f"Degradation rate: {degraded / total:.4f}")

    print("\n========== EXAMPLE FAILURES PER TRANSFORMATION ==========")

    for transform in sorted(by_transform):
        failures = by_transform[transform]["clean_right_perturbed_wrong"]
        print(f"\nTransformation: {transform}")
        for row in failures[:3]:
            print("Original:", row["original_sentence"])
            print("Modified:", row["modified_sentence"])
            print("Original label:", row["original_label"], "| Modified label:", row["modified_label"])
            print("Modified prediction:", row["modified_prediction"])
            print()


def main():
    """Run the selected analysis mode."""
    args = parse_args()
    rows = load_rows(args.predictions)
    print(f"Loaded {len(rows)} prediction rows from {args.predictions}")

    if args.mode == "intensity":
        run_intensity(rows)
    else:
        run_errors(rows)


if __name__ == "__main__":
    main()
