
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = (
    ROOT_DIR
    / "data"
    / "raw"
    / "Womens Clothing E-Commerce Reviews.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "reviews_cleaned.csv"


def load_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def clean_data(df):

    initial_rows = len(df)

    # Remove unnamed index columns
    df = df.loc[
        :,
        ~df.columns.str.startswith("unnamed")
    ].copy()

    # Remove rows without review text
    df["review_text"] = df["review_text"].fillna("")

    df["title"] = df["title"].fillna("")

    # Combine title and review
    df["full_review"] = (
        df["title"].astype(str).str.strip()
        + " "
        + df["review_text"].astype(str).str.strip()
    ).str.strip()

    # Remove empty reviews
    df = df[df["full_review"].str.len() > 0].copy()

    # Normalize review text for duplicate detection
    df["normalized_review"] = (
        df["full_review"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    duplicate_count = df["normalized_review"].duplicated().sum()

    df = df.drop_duplicates(
        subset=["normalized_review"]
    ).copy()

    # Review metrics
    df["review_length"] = df["full_review"].str.len()

    df["word_count"] = (
        df["full_review"].str.split().str.len()
    )

    df["has_title"] = df["title"].str.len() > 0

    # Rating-based sentiment proxy labels
    df["sentiment"] = df["rating"].map({
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive"
    })

    print("\nDATA CLEANING REPORT")
    print("-" * 40)

    print(f"Initial rows: {initial_rows}")
    print(f"Duplicate reviews: {duplicate_count}")
    print(f"Final rows: {len(df)}")
    print(f"Removed rows: {initial_rows - len(df)}")

    return df


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    print("\nOriginal dataset shape:", df.shape)

    cleaned_df = clean_data(df)

    cleaned_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved successfully!")
    print("Output:", OUTPUT_FILE)

    print("\nFinal dataset shape:", cleaned_df.shape)

    print("\nSample records:")
    print(
        cleaned_df[
            ["title", "review_text", "rating", "sentiment"]
        ].head()
    )


if __name__ == "__main__":
    main()
 