from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

REVIEWS_FILE = ROOT_DIR / "data" / "processed" / "reviews_quality_scored.csv"
SIMILAR_FILE = ROOT_DIR / "data" / "processed" / "similar_reviews.csv"

ASPECT_DIR = ROOT_DIR / "assets" / "aspects"
EDA_DIR = ROOT_DIR / "assets" / "eda"
MODEL_METRICS_FILE = ROOT_DIR / "assets" / "models" / "sentiment_metrics.json"

st.set_page_config(
    page_title="ReviewIQ | Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 15px;
            border-radius: 10px;
        }
        .small-note {
            color: #888888;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_reviews():
    if not REVIEWS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(REVIEWS_FILE)

    # Normalize selected text columns
    for col in [
        "sentiment",
        "quality_label",
        "department_name",
        "division_name",
        "class_name",
        "detected_aspects",
        "full_review",
        "review_text",
        "title",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


@st.cache_data
def load_similar_reviews():
    if not SIMILAR_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(SIMILAR_FILE)


@st.cache_data
def load_aspect_data(filename):
    path = ASPECT_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_model_metrics():
    if not MODEL_METRICS_FILE.exists():
        return {}

    try:
        import json

        with open(MODEL_METRICS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


df = load_reviews()
similar_df = load_similar_reviews()

if df.empty:
    st.error(
        "Could not load the cleaned review dataset. "
        "Check that data/processed/reviews_quality_scored.csv exists."
    )
    st.stop()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def format_percentage(value):
    return f"{value:.1f}%"


def get_review_text(row):
    title = str(row.get("title", "")).strip()
    review = str(row.get("review_text", "")).strip()

    if title and title.lower() != "nan":
        return f"{title} — {review}" if review else title

    return review or str(row.get("full_review", ""))


def split_aspects(value):
    if pd.isna(value):
        return []

    return [
        aspect.strip().lower()
        for aspect in str(value).split(",")
        if aspect.strip()
    ]


def find_column(columns, possible_names):
    """Find a column using a list of likely column names."""
    normalized = {str(col).lower().strip(): col for col in columns}

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def show_empty_message(message):
    st.info(message)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("📊 ReviewIQ")
st.sidebar.caption("AI-Powered Review Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sentiment Analysis",
        "Aspect Intelligence",
        "Review Quality",
        "Similar Reviews",
        "Review Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.caption("Dataset Filters")

filtered_df = df.copy()

if "rating" in filtered_df.columns:
    ratings = sorted(filtered_df["rating"].dropna().unique().tolist())

    selected_ratings = st.sidebar.multiselect(
        "Rating",
        ratings,
        default=ratings,
    )

    filtered_df = filtered_df[
        filtered_df["rating"].isin(selected_ratings)
    ]

if "sentiment" in filtered_df.columns:
    sentiments = sorted(
        value
        for value in filtered_df["sentiment"].dropna().unique()
        if value
    )

    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        sentiments,
        default=sentiments,
    )

    filtered_df = filtered_df[
        filtered_df["sentiment"].isin(selected_sentiments)
    ]

if "department_name" in filtered_df.columns:

    filtered_df["department_name"] = (
        filtered_df["department_name"]
        .fillna("Unspecified")
        .replace(r"^\s*$", "Unspecified", regex=True)
    )

    departments = sorted(
        filtered_df["department_name"].unique().tolist()
    )

    selected_departments = st.sidebar.multiselect(
        "Department",
        departments,
        default=departments,
    )

    filtered_df = filtered_df[
        filtered_df["department_name"].isin(selected_departments)
    ]


if "quality_label" in filtered_df.columns:
    filtered_df["quality_label"] = (
        filtered_df["quality_label"].replace("", "Unspecified")
    )

    quality_labels = sorted(filtered_df["quality_label"].unique().tolist())

    selected_quality = st.sidebar.multiselect(
        "Review Quality",
        quality_labels,
        default=quality_labels,
    )

    filtered_df = filtered_df[
        filtered_df["quality_label"].isin(selected_quality)
    ]
    
    
st.sidebar.divider()
st.sidebar.caption(f"Showing {len(filtered_df):,} of {len(df):,} reviews")


# ---------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

    st.title("Review Intelligence Dashboard")
    st.markdown(
        "Explore customer feedback, product sentiment, "
        "review quality and apparel-related insights."
    )

    st.divider()

    total_reviews = len(filtered_df)

    avg_rating = (
        filtered_df["rating"].mean()
        if "rating" in filtered_df.columns and total_reviews
        else 0
    )

    avg_quality = (
        filtered_df["quality_score"].mean()
        if "quality_score" in filtered_df.columns and total_reviews
        else 0
    )

    recommendation_rate = (
        filtered_df["recommended_ind"].mean() * 100
        if "recommended_ind" in filtered_df.columns and total_reviews
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Average Rating", f"{avg_rating:.2f} / 5")
    col3.metric("Recommendation Rate", format_percentage(recommendation_rate))
    col4.metric("Average Quality Score", f"{avg_quality:.1f} / 100")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Rating Distribution")

        if "rating" in filtered_df.columns:
            rating_data = (
                filtered_df["rating"]
                .value_counts()
                .sort_index()
                .rename_axis("Rating")
                .reset_index(name="Reviews")
            )

            fig = px.bar(
                rating_data,
                x="Rating",
                y="Reviews",
                text="Reviews",
                color="Rating",
                title="Customer Ratings",
            )

            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty_message("Rating data is unavailable.")

    with right:
        st.subheader("Sentiment Distribution")

        if "sentiment" in filtered_df.columns:
            sentiment_data = (
                filtered_df["sentiment"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Reviews")
            )

            fig = px.pie(
                sentiment_data,
                names="Sentiment",
                values="Reviews",
                hole=0.45,
                title="Customer Sentiment",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty_message("Sentiment data is unavailable.")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Department-wise Reviews")

        if "department_name" in filtered_df.columns:
            department_data = (
                filtered_df["department_name"]
                .value_counts()
                .reset_index()
            )
            department_data.columns = ["Department", "Reviews"]

            fig = px.bar(
                department_data,
                x="Department",
                y="Reviews",
                text="Reviews",
                title="Reviews by Department",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty_message("Department data is unavailable.")

    with right:
        st.subheader("Review Quality Distribution")

        if "quality_label" in filtered_df.columns:
            quality_data = (
                filtered_df["quality_label"]
                .value_counts()
                .rename_axis("Quality")
                .reset_index(name="Reviews")
            )

            fig = px.pie(
                quality_data,
                names="Quality",
                values="Reviews",
                hole=0.45,
                title="Quality Categories",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty_message("Quality labels are unavailable.")

    st.divider()

    st.subheader("Key Dataset Insights")

    if not filtered_df.empty:
        if "sentiment" in filtered_df.columns:
            most_common_sentiment = filtered_df["sentiment"].mode()

            if not most_common_sentiment.empty:
                st.write(
                    f"- Most common sentiment: **{most_common_sentiment.iloc[0].title()}**"
                )

        if "detected_aspects" in filtered_df.columns:
            aspect_counts = {}

            for value in filtered_df["detected_aspects"]:
                for aspect in split_aspects(value):
                    aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

            if aspect_counts:
                top_aspect = max(aspect_counts, key=aspect_counts.get)
                st.write(
                    f"- Most discussed aspect: **{top_aspect.title()}**"
                )

        if "quality_label" in filtered_df.columns:
            low_count = (
                filtered_df["quality_label"].str.lower() == "low"
            ).sum()

            st.write(f"- Reviews classified as low quality: **{low_count:,}**")

    st.caption(
        "Note: Review quality is a heuristic score based on review detail, "
        "aspects, experience cues and specificity. It is not a verified "
        "measure of review authenticity."
    )


# ---------------------------------------------------------
# PAGE: SENTIMENT ANALYSIS
# ---------------------------------------------------------

elif page == "Sentiment Analysis":

    st.title("Sentiment Analysis")
    st.markdown(
        "Understand how customers express positive, neutral and negative "
        "opinions about apparel products."
    )

    if "sentiment" not in filtered_df.columns:
        show_empty_message("Sentiment column is unavailable.")
        st.stop()

    sentiment_counts = (
        filtered_df["sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    total = sentiment_counts["Reviews"].sum()

    if total:
        sentiment_counts["Percentage"] = (
            sentiment_counts["Reviews"] / total * 100
        )

    col1, col2, col3 = st.columns(3)

    for col, label in zip(
        [col1, col2, col3],
        ["positive", "neutral", "negative"],
    ):
        count = (
            int(
                filtered_df["sentiment"]
                .str.lower()
                .eq(label)
                .sum()
            )
        )

        percentage = count / len(filtered_df) * 100 if len(filtered_df) else 0

        col.metric(
            f"{label.title()} Reviews",
            f"{count:,}",
            format_percentage(percentage),
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Reviews",
            hole=0.4,
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Sentiment by Rating")

        if "rating" in filtered_df.columns:
            cross_tab = pd.crosstab(
                filtered_df["rating"],
                filtered_df["sentiment"],
            ).reset_index()

            fig = px.bar(
                cross_tab,
                x="rating",
                y=[
                    col for col in cross_tab.columns
                    if col != "rating"
                ],
                barmode="group",
                labels={
                    "rating": "Rating",
                    "value": "Reviews",
                    "variable": "Sentiment",
                },
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            show_empty_message("Rating data is unavailable.")

    st.divider()

    st.subheader("Sentiment Model Performance")

    metrics = load_model_metrics()

    if metrics:
        st.json(metrics)
    else:
        st.info(
            "Model metrics JSON was not found. "
            "The trained model's metrics can be displayed here when available."
        )

    st.caption(
        "Sentiment labels are derived from customer ratings and used as "
        "proxy labels for model training. Model predictions are not "
        "independently verified emotional assessments."
    )


# ---------------------------------------------------------
# PAGE: ASPECT INTELLIGENCE
# ---------------------------------------------------------

elif page == "Aspect Intelligence":

    st.title("Apparel Aspect Intelligence")
    st.markdown(
        "Identify the product attributes customers discuss most frequently "
        "and examine their associated sentiment."
    )

    if "detected_aspects" not in filtered_df.columns:
        show_empty_message("Aspect data is unavailable.")
        st.stop()

    aspect_rows = []

    for _, row in filtered_df.iterrows():
        aspects = split_aspects(row.get("detected_aspects", ""))

        for aspect in aspects:
            aspect_rows.append(
                {
                    "Aspect": aspect.title(),
                    "Sentiment": row.get("sentiment", "unknown"),
                    "Quality": row.get("quality_label", "unknown"),
                    "Rating": row.get("rating", None),
                }
            )

    aspect_df = pd.DataFrame(aspect_rows)

    if aspect_df.empty:
        show_empty_message("No detected aspects are available.")
        st.stop()

    aspect_counts = (
        aspect_df["Aspect"]
        .value_counts()
        .rename_axis("Aspect")
        .reset_index(name="Mentions")
    )

    st.subheader("Most Discussed Product Aspects")

    fig = px.bar(
        aspect_counts.sort_values("Mentions"),
        x="Mentions",
        y="Aspect",
        orientation="h",
        text="Mentions",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment by Aspect")

        sentiment_by_aspect = (
            aspect_df.groupby(["Aspect", "Sentiment"])
            .size()
            .reset_index(name="Reviews")
        )

        fig = px.bar(
            sentiment_by_aspect,
            x="Aspect",
            y="Reviews",
            color="Sentiment",
            barmode="group",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Rating by Aspect")

        rating_by_aspect = (
            aspect_df.dropna(subset=["Rating"])
            .groupby("Aspect")["Rating"]
            .mean()
            .reset_index(name="Average Rating")
        )

        fig = px.bar(
            rating_by_aspect,
            x="Aspect",
            y="Average Rating",
            text_auto=".2f",
        )

        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Aspect Summary")

    summary = (
        aspect_df.groupby("Aspect")
        .agg(
            Mentions=("Aspect", "size"),
            Average_Rating=("Rating", "mean"),
        )
        .reset_index()
    )

    summary["Average_Rating"] = summary["Average_Rating"].round(2)

    st.dataframe(
        summary.sort_values("Mentions", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Aspect counts are mentions, not unique reviews. "
        "A single review can mention multiple product aspects."
    )


# ---------------------------------------------------------
# PAGE: REVIEW QUALITY
# ---------------------------------------------------------

elif page == "Review Quality":

    st.title("Review Quality Analysis")
    st.markdown(
        "Examine review detail and the heuristic quality scores "
        "assigned by the ReviewIQ scoring system."
    )

    required = ["quality_score", "quality_label"]

    if not all(col in filtered_df.columns for col in required):
        show_empty_message("Quality score columns are unavailable.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Quality Score",
        f"{filtered_df['quality_score'].mean():.2f}"
        if len(filtered_df) else "0",
    )

    col2.metric(
        "High Quality Reviews",
        f"{(filtered_df['quality_label'].str.lower() == 'high').sum():,}",
    )

    col3.metric(
        "Average Review Length",
        f"{filtered_df['word_count'].mean():.1f} words"
        if "word_count" in filtered_df.columns and len(filtered_df)
        else "N/A",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Quality Category Distribution")

        quality_counts = (
            filtered_df["quality_label"]
            .value_counts()
            .rename_axis("Quality")
            .reset_index(name="Reviews")
        )

        fig = px.pie(
            quality_counts,
            names="Quality",
            values="Reviews",
            hole=0.4,
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Quality Score Distribution")

        fig = px.histogram(
            filtered_df,
            x="quality_score",
            nbins=20,
            title="Review Quality Scores",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    score_columns = [
        "detail_score",
        "aspect_score",
        "experience_score",
        "specificity_score",
    ]

    available_score_columns = [
        col for col in score_columns
        if col in filtered_df.columns
    ]

    if available_score_columns:
        st.subheader("Quality Score Components")

        component_data = (
            filtered_df[available_score_columns]
            .mean()
            .rename_axis("Component")
            .reset_index(name="Average Score")
        )

        component_data["Component"] = (
            component_data["Component"]
            .str.replace("_score", "", regex=False)
            .str.replace("_", " ")
            .str.title()
        )

        fig = px.bar(
            component_data,
            x="Component",
            y="Average Score",
            text_auto=".2f",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Example Reviews")

    quality_choice = st.selectbox(
        "Select quality category",
        ["high", "medium", "low"],
    )

    examples = filtered_df[
        filtered_df["quality_label"].str.lower() == quality_choice
    ].sort_values("quality_score", ascending=False)

    if examples.empty:
        show_empty_message("No reviews found for this quality category.")
    else:
        display_columns = [
            col for col in [
                "rating",
                "sentiment",
                "quality_score",
                "quality_label",
                "word_count",
                "detected_aspects",
                "full_review",
            ]
            if col in examples.columns
        ]

        st.dataframe(
            examples[display_columns].head(20),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Quality scores are heuristic indicators, not human-verified "
        "quality labels or proof that a review is genuine."
    )


# ---------------------------------------------------------
# PAGE: SIMILAR REVIEWS
# ---------------------------------------------------------

elif page == "Similar Reviews":

    st.title("Similar Review Detection")
    st.markdown(
        "Explore review pairs flagged by TF-IDF and cosine similarity. "
        "These are candidates for manual inspection, not confirmed duplicates."
    )

    if similar_df.empty:
        show_empty_message(
            "Similar review data was not found. "
            "Run src/analysis/duplicate_detection.py first."
        )
        st.stop()

    st.metric("Flagged Similar Pairs", f"{len(similar_df):,}")

    similarity_col = find_column(
        similar_df.columns,
        ["similarity", "similarity_score", "cosine_similarity", "score"],
    )

    if similarity_col:
        similarity_values = pd.to_numeric(
            similar_df[similarity_col],
            errors="coerce",
        ).dropna()

        if not similarity_values.empty:
            st.metric(
                "Average Similarity",
                f"{similarity_values.mean():.3f}",
            )

            fig = px.histogram(
                pd.DataFrame({"Similarity": similarity_values}),
                x="Similarity",
                nbins=15,
                title="Similarity Score Distribution",
            )

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Flagged Review Pairs")

    st.dataframe(
        similar_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "High similarity does not prove plagiarism, manipulation or "
        "duplicate submission. Similar pairs should be reviewed manually."
    )


# ---------------------------------------------------------
# PAGE: REVIEW EXPLORER
# ---------------------------------------------------------

elif page == "Review Explorer":

    st.title("Review Explorer")
    st.markdown(
        "Search and inspect individual customer reviews using the "
        "dataset filters in the sidebar."
    )

    search_text = st.text_input(
        "Search review text",
        placeholder="Search words such as fit, fabric, comfort...",
    )

    explorer_df = filtered_df.copy()

    if search_text.strip():
        search_columns = [
            col for col in ["full_review", "title", "review_text"]
            if col in explorer_df.columns
        ]

        if search_columns:
            search_mask = pd.Series(
                False,
                index=explorer_df.index,
            )

            for col in search_columns:
                search_mask |= explorer_df[col].str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False,
                )

            explorer_df = explorer_df[search_mask]

    if "detected_aspects" in explorer_df.columns:
        available_aspects = sorted(
            {
                aspect
                for value in explorer_df["detected_aspects"]
                for aspect in split_aspects(value)
            }
        )

        selected_aspect = st.selectbox(
            "Filter by product aspect",
            ["All"] + available_aspects,
        )

        if selected_aspect != "All":
            explorer_df = explorer_df[
                explorer_df["detected_aspects"].apply(
                    lambda value: selected_aspect
                    in split_aspects(value)
                )
            ]

    st.write(f"**Matching reviews:** {len(explorer_df):,}")

    display_columns = [
        col for col in [
            "clothing_id",
            "rating",
            "sentiment",
            "recommended_ind",
            "department_name",
            "class_name",
            "quality_score",
            "quality_label",
            "detected_aspects",
            "full_review",
        ]
        if col in explorer_df.columns
    ]

    st.dataframe(
        explorer_df[display_columns],
        use_container_width=True,
        hide_index=True,
        height=550,
    )

    st.download_button(
        "Download filtered reviews as CSV",
        data=explorer_df.to_csv(index=False).encode("utf-8"),
        file_name="reviewiq_filtered_reviews.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "ReviewIQ | AI-Powered Review Intelligence System | "
    "Built with Python, Pandas, Scikit-learn, Plotly and Streamlit"
)