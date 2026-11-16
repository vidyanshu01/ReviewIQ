
from pathlib import Path

import pandas as pd
import plotly.express as px


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR / "data" / "processed" / "reviews_quality_scored.csv"
)

OUTPUT_DIR = ROOT_DIR / "assets" / "aspects"


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    df["detected_aspects"] = (
        df["detected_aspects"].fillna("")
    )

    # Convert comma-separated aspects into individual rows
    aspect_df = df.assign(
        aspect=df["detected_aspects"].str.split(", ")
    ).explode("aspect")

    aspect_df = aspect_df[
        aspect_df["aspect"].str.len() > 0
    ].copy()

    # Overall aspect frequency
    frequency = (
        aspect_df["aspect"]
        .value_counts()
        .rename_axis("aspect")
        .reset_index(name="review_count")
    )

    frequency.to_csv(
        OUTPUT_DIR / "aspect_frequency.csv",
        index=False
    )

    # Sentiment distribution for each aspect
    sentiment_analysis = pd.crosstab(
        aspect_df["aspect"],
        aspect_df["sentiment"]
    )

    sentiment_analysis.to_csv(
        OUTPUT_DIR / "aspect_sentiment.csv"
    )

    # Quality distribution for each aspect
    quality_analysis = pd.crosstab(
        aspect_df["aspect"],
        aspect_df["quality_label"]
    )

    quality_analysis.to_csv(
        OUTPUT_DIR / "aspect_quality.csv"
    )

    # Visualization 1: Aspect frequency
    fig1 = px.bar(
        frequency.sort_values("review_count"),
        x="review_count",
        y="aspect",
        orientation="h",
        title="Most Discussed Apparel Attributes",
        labels={
            "review_count": "Number of Reviews",
            "aspect": "Product Attribute"
        }
    )

    fig1.write_html(
        OUTPUT_DIR / "aspect_frequency.html"
    )

    # Visualization 2: Aspect sentiment
    sentiment_long = (
        aspect_df.groupby(["aspect", "sentiment"])
        .size()
        .reset_index(name="review_count")
    )

    fig2 = px.bar(
        sentiment_long,
        x="aspect",
        y="review_count",
        color="sentiment",
        barmode="group",
        title="Customer Sentiment by Product Attribute"
    )

    fig2.write_html(
        OUTPUT_DIR / "aspect_sentiment.html"
    )

    print("\nASPECT INTELLIGENCE REPORT")
    print("=" * 45)

    print("\nAspect frequency:")
    print(frequency.to_string(index=False))

    print("\nSentiment by aspect:")
    print(sentiment_analysis)

    print("\nQuality by aspect:")
    print(quality_analysis)

    print("\nReports saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
