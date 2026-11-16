
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from src.nlp.text_preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT_DIR / "data" / "processed" / "reviews_cleaned.csv"
)

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "assets" / "models"

MODEL_FILE = MODEL_DIR / "sentiment_model.joblib"


def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required_columns = ["full_review", "sentiment"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    df = df.dropna(
        subset=["full_review", "sentiment"]
    ).copy()

    df["cleaned_text"] = df["full_review"].apply(
        clean_text
    )

    df = df[df["cleaned_text"].str.len() > 0]

    return df


def train_model(df):

    X = df["cleaned_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("\nTraining sentiment model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"]
    )

    print("\nMODEL EVALUATION")
    print("=" * 50)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    return model, {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": ["negative", "neutral", "positive"],
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    print("Dataset shape:", df.shape)
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    model, metrics = train_model(df)

    joblib.dump(model, MODEL_FILE)

    with open(
        REPORT_DIR / "sentiment_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nModel saved:", MODEL_FILE)
    print("Metrics saved:", REPORT_DIR / "sentiment_metrics.json")


if __name__ == "__main__":
    main()
