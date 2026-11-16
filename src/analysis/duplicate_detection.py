
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR / "data" / "processed" / "reviews_quality_scored.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "similar_reviews.csv"

SIMILARITY_THRESHOLD = 0.85


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    df["full_review"] = (
        df["full_review"].fillna("").astype(str)
    )

    return df


def detect_duplicates(df):

    texts = df["full_review"].tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(texts)

    print("TF-IDF matrix shape:", matrix.shape)
    print("Finding similar reviews...")

    # Cosine distance = 1 - cosine similarity
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        radius=1 - SIMILARITY_THRESHOLD,
        n_jobs=-1
    )

    model.fit(matrix)

    distances, indices = model.radius_neighbors(
        matrix,
        return_distance=True,
        sort_results=True
    )

    pairs = []

    for i, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):

        for distance, j in zip(row_distances, row_indices):

            # Exclude self-comparison and reverse pairs
            if j <= i:
                continue

            similarity = 1 - distance

            pairs.append({
                "review_1_index": int(i),
                "review_2_index": int(j),
                "similarity": round(float(similarity), 4),
                "review_1": texts[i],
                "review_2": texts[j]
            })

    return pd.DataFrame(pairs)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    print("Total reviews:", len(df))

    similar_reviews = detect_duplicates(df)

    similar_reviews.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDUPLICATE DETECTION REPORT")
    print("=" * 45)

    print(
        "Similarity threshold:",
        SIMILARITY_THRESHOLD
    )

    print(
        "Similar review pairs:",
        len(similar_reviews)
    )

    if not similar_reviews.empty:

        print("\nTop 10 similar pairs:")

        top_pairs = similar_reviews.nlargest(
            10,
            "similarity"
        )

        for _, row in top_pairs.iterrows():

            print("\nSimilarity:", row["similarity"])

            print("Review 1:", row["review_1"][:200])
            print("Review 2:", row["review_2"][:200])

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
