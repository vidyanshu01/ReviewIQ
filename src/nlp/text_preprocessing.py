
import re
import html


def clean_text(text: str) -> str:
    """
    Normalize review text while preserving negations
    such as 'not', 'no', and 'never'.
    """

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    samples = [
        "This dress is AMAZING!",
        "The fabric is NOT comfortable.",
        "Good quality & beautiful design."
    ]

    for sample in samples:
        print("Original:", sample)
        print("Cleaned:", clean_text(sample))
        print("-" * 40)
