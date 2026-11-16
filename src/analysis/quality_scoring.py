
from pathlib import Path
import re

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "reviews_cleaned.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "reviews_quality_scored.csv"

ASPECTS = {
    "fit": [
        "fit", "fitting", "size", "sizing", "tight",
        "loose", "small", "large", "length", "waist"
    ],
    "fabric": [
        "fabric", "material", "cotton", "silk",
        "polyester", "linen", "texture", "quality"
    ],
    "comfort": [
        "comfortable", "comfort", "soft", "itchy",
        "scratchy", "breathable", "stretch"
    ],
    "design": [
        "design", "style", "color", "colour",
        "pattern", "print", "look", "appearance"
    ],
    "durability": [
        "durable", "durability", "washing", "washed",
        "shrink", "shrunk", "faded", "stitching",
        "seam", "pilling"
    ]
}

EXPERIENCE_CUES = [
    "i wore", "i washed", "after washing",
    "after wearing", "i ordered", "i bought",
    "i received", "when i wore", "when i washed",
    "i returned", "i purchased", "came apart",
    "after one wash", "after several washes"
]

SPECIFICITY_CUES = [
    "true to size", "runs small", "runs large",
    "too short", "too long", "too tight",
    "too loose", "see through", "after one wash",
    "after several washes", "fell apart",
    "lost its shape", "color faded",
    "shrunk after washing", "stitching came apart",
    "poor stitching", "excellent stitching",
    "fits perfectly", "one size larger",
    "one size smaller"
]


def contains_keyword(text, keyword):
    return bool(
        re.search(
            rf"\b{re.escape(keyword)}\b",
            text
        )
    )


def analyze_review(review):

    text = str(review).lower().strip()
    word_count = len(text.split())

    # 1. Detail score: maximum 20
    if word_count <= 5:
        detail_score = 3
    elif word_count <= 15:
        detail_score = 8
    elif word_count <= 30:
        detail_score = 13
    elif word_count <= 60:
        detail_score = 17
    else:
        detail_score = 20

    # 2. Product aspect coverage: maximum 30
    detected_aspects = []

    for aspect, keywords in ASPECTS.items():
        if any(contains_keyword(text, k) for k in keywords):
            detected_aspects.append(aspect)

    aspect_score = min(len(detected_aspects) * 7, 30)

    # 3. Customer experience: maximum 25
    experience_matches = sum(
        cue in text for cue in EXPERIENCE_CUES
    )

    experience_score = min(
        experience_matches * 8, 25
    )

    # 4. Specificity: maximum 25
    specificity_matches = sum(
        cue in text for cue in SPECIFICITY_CUES
    )

    specificity_score = min(
        specificity_matches * 7, 20
    )

    has_measurement = bool(
        re.search(r"\b\d+(?:\.\d+)?\b", text)
    )

    if has_measurement:
        specificity_score = min(
            specificity_score + 5, 25
        )

    total_score = (
        detail_score
        + aspect_score
        + experience_score
        + specificity_score
    )

    total_score = min(total_score, 100)

    if total_score >= 70:
        quality_label = "high"
    elif total_score >= 40:
        quality_label = "medium"
    else:
        quality_label = "low"

    return {
        "quality_score": total_score,
        "quality_label": quality_label,
        "detail_score": detail_score,
        "aspect_score": aspect_score,
        "experience_score": experience_score,
        "specificity_score": specificity_score,
        "aspect_count": len(detected_aspects),
        "detected_aspects": ", ".join(detected_aspects)
    }


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    results = df["full_review"].apply(analyze_review)

    results_df = pd.DataFrame(results.tolist())

    df = pd.concat(
        [df.reset_index(drop=True), results_df],
        axis=1
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nUPDATED QUALITY REPORT")
    print("=" * 45)

    print(df["quality_label"].value_counts())

    print("\nAverage quality:", round(
        df["quality_score"].mean(), 2
    ))

    print("\nAverage aspect count:", round(
        df["aspect_count"].mean(), 2
    ))

    print("\nUpdated file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
