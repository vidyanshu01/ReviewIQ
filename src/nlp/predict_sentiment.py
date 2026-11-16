
from pathlib import Path

import joblib

from src.nlp.text_preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = ROOT_DIR / "models" / "sentiment_model.joblib"


def predict_sentiment(review: str):

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Train the sentiment model first."
        )

    model = joblib.load(MODEL_FILE)

    cleaned = clean_text(review)

    if not cleaned:
        raise ValueError("Review cannot be empty.")

    prediction = model.predict([cleaned])[0]

    probabilities = model.predict_proba([cleaned])[0]

    confidence = max(probabilities)

    return prediction, confidence


if __name__ == "__main__":

    test_reviews = [
        "I absolutely love this dress. The fabric is beautiful!",
        "Terrible quality. The material is uncomfortable.",
        "The product is okay, nothing special."
    ]

    for review in test_reviews:

        sentiment, confidence = predict_sentiment(review)

        print("\nReview:", review)
        print("Predicted sentiment:", sentiment)
        print(f"Confidence: {confidence:.2%}")
