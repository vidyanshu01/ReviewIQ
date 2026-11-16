
from pathlib import Path

import pandas as pd
import plotly.express as px


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT_DIR / "data" / "processed" / "reviews_cleaned.csv"
)

OUTPUT_DIR = ROOT_DIR / "assets" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    return pd.read_csv(DATA_FILE)


def analyze_missing_values(df):
    print("\n1. MISSING VALUE ANALYSIS")
    print("-" * 45)

    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    print(missing.sort_values(
        "missing_percentage",
        ascending=False
    ))

    missing.to_csv(
        OUTPUT_DIR / "missing_values.csv"
    )


def analyze_ratings(df):
    print("\n2. RATING DISTRIBUTION")
    print("-" * 45)

    print(df["rating"].value_counts().sort_index())

    fig = px.histogram(
        df,
        x="rating",
        color="rating",
        title="Customer Rating Distribution",
        labels={"rating": "Star Rating"},
        text_auto=True
    )

    fig.write_html(
        OUTPUT_DIR / "rating_distribution.html"
    )


def analyze_sentiment(df):
    print("\n3. SENTIMENT DISTRIBUTION")
    print("-" * 45)

    sentiment = df["sentiment"].value_counts()

    print(sentiment)

    fig = px.pie(
        names=sentiment.index,
        values=sentiment.values,
        title="Review Sentiment Distribution",
        hole=0.4
    )

    fig.write_html(
        OUTPUT_DIR / "sentiment_distribution.html"
    )


def analyze_categories(df):
    print("\n4. PRODUCT CATEGORY ANALYSIS")
    print("-" * 45)

    for column in [
        "department_name",
        "class_name",
        "division_name"
    ]:
        if column in df.columns:
            print(f"\n{column}")
            print(df[column].value_counts().head(10))

    if "department_name" in df.columns:
        fig = px.histogram(
            df,
            y="department_name",
            color="sentiment",
            title="Reviews by Department",
            barmode="group"
        )

        fig.write_html(
            OUTPUT_DIR / "department_analysis.html"
        )


def analyze_review_lengths(df):
    print("\n5. REVIEW LENGTH ANALYSIS")
    print("-" * 45)

    print(df["word_count"].describe())

    fig = px.histogram(
        df,
        x="word_count",
        nbins=50,
        title="Distribution of Review Word Counts",
        labels={"word_count": "Number of Words"}
    )

    fig.write_html(
        OUTPUT_DIR / "review_length_distribution.html"
    )


def analyze_recommendations(df):
    print("\n6. CUSTOMER RECOMMENDATIONS")
    print("-" * 45)

    print(df["recommended_ind"].value_counts())

    recommendation = (
        df.groupby("rating")["recommended_ind"]
        .mean()
        .reset_index()
    )

    print("\nRecommendation rate by rating:")
    print(recommendation)

    fig = px.bar(
        recommendation,
        x="rating",
        y="recommended_ind",
        title="Recommendation Rate by Rating",
        labels={
            "rating": "Star Rating",
            "recommended_ind": "Recommendation Rate"
        }
    )

    fig.write_html(
        OUTPUT_DIR / "recommendation_analysis.html"
    )


def main():

    df = load_data()

    print("=" * 55)
    print("             REVIEWIQ - EDA REPORT")
    print("=" * 55)

    print("\nDataset shape:", df.shape)

    analyze_missing_values(df)
    analyze_ratings(df)
    analyze_sentiment(df)
    analyze_categories(df)
    analyze_review_lengths(df)
    analyze_recommendations(df)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print("Reports saved at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
