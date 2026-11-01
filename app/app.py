"""
IMDB Movie Review Sentiment Analyzer (Streamlit app)

Type or paste a movie review and get a positive / negative prediction with a
confidence score. The text goes through the same cleaning steps as in the
notebook, then through the saved vectorizer + model pipeline.

Run from the repository root:
    streamlit run app/app.py
"""
import re
import warnings
from pathlib import Path

import joblib
import nltk
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

# BeautifulSoup warns on very short inputs that look like file names or URLs
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
POSITIVE_LABEL = "positive"
THRESHOLD = 0.5  # probability of "positive" at or above this value -> positive review

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# The calibrated pipeline is preferred because it outputs real probabilities.
# The raw pipeline (Linear SVC) is the fallback.
MODEL_FILES = [
    "sentiment_pipeline_calibrated.joblib",
    "final_sentiment_pipeline.joblib",
]

EXAMPLES = {
    "Positive example": (
        "An absolute masterpiece. The acting was superb, the story kept me hooked "
        "from start to finish, and the ending was perfect."
    ),
    "Negative example": (
        "What a waste of time. The plot made no sense, the acting was terrible "
        "and I wanted to leave halfway through."
    ),
    "Negation example": "The movie was not good at all. I expected much more.",
}

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")


# ------------------------------------------------------------------
# Loading (cached, so it runs only once per server session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the first available saved pipeline from the models folder."""
    for file_name in MODEL_FILES:
        path = MODELS_DIR / file_name
        if path.exists():
            return joblib.load(path), file_name
    return None, None


@st.cache_resource
def load_text_tools():
    """Tokenizer and stopword list, identical to the notebook (negations are kept)."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    negations = {"not", "no", "nor", "never", "don't", "didn't", "isn't", "wasn't"}
    stop_words = set(stopwords.words("english")) - negations
    return ToktokTokenizer(), stop_words


# ------------------------------------------------------------------
# Text cleaning: same steps and order as the notebook
# ------------------------------------------------------------------
def clean_review(text, tokenizer, stop_words):
    text = text.replace("\u2019", "'")                      # curly apostrophe -> straight (pasted text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")  # 1. strip HTML tags
    text = re.sub(r"\[[^]]*\]", "", text)                   # 2. remove [bracketed] text
    text = text.replace("'", "")                            # 3. don't -> dont
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)             # 4. remove special characters
    tokens = [t.strip() for t in tokenizer.tokenize(text.lower())]   # 5. lowercase + tokenize
    return " ".join(t for t in tokens if t and t not in stop_words)  # 6. remove stopwords


# ------------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------------
def has_known_words(model, cleaned_text):
    """False if none of the words exist in the vectorizer vocabulary."""
    return model[:-1].transform([cleaned_text]).nnz > 0


def positive_probability(model, cleaned_text):
    """Return (probability of the positive class, whether it is a calibrated probability)."""
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        pos_idx = classes.index(POSITIVE_LABEL)
        return float(model.predict_proba([cleaned_text])[0][pos_idx]), True

    # Raw Linear SVC: convert the decision score to a 0-1 value with a sigmoid
    score = float(model.decision_function([cleaned_text])[0])  # score of classes_[1]
    if classes[1] != POSITIVE_LABEL:
        score = -score
    return float(1 / (1 + np.exp(-score))), False


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
model, model_file = load_model()
tokenizer, stop_words = load_text_tools()

if model is None:
    st.error(
        "No model file found. Put `final_sentiment_pipeline.joblib` "
        "(or `sentiment_pipeline_calibrated.joblib`) in the `models/` folder."
    )
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies a movie review as **positive** or **negative**.\n\n"
        "**Model:** TF-IDF (1-2 grams) + Linear SVC, trained on IMDB reviews. "
        "About 0.91 accuracy and F1 on a held-out test set."
    )
    st.write(
        "**How it works**\n"
        "1. The review is cleaned (HTML, special characters, stopwords; negations kept).\n"
        "2. The text is converted to TF-IDF features.\n"
        "3. The classifier predicts the sentiment."
    )
    st.write(
        "**Limitations**\n"
        "- Sarcasm and mixed reviews are often misclassified.\n"
        "- Trained on movie reviews only, so other domains may not work well.\n"
        "- Works with English text only."
    )
    st.caption(f"Loaded model file: {model_file}")

st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Write or paste a movie review in English and the model will predict whether it is positive or negative.")

st.session_state.setdefault("review_text", "")


def set_example(text):
    st.session_state["review_text"] = text


example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES.items()):
    col.button(label, on_click=set_example, args=(text,))

review = st.text_area(
    "Your review",
    key="review_text",
    height=180,
    placeholder="Type or paste a movie review here...",
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_review(review, tokenizer, stop_words)

        if not cleaned or not has_known_words(model, cleaned):
            st.warning(
                "The review has too few recognizable words to make a prediction. "
                "Try writing a longer review in English."
            )
        else:
            p_pos, is_calibrated = positive_probability(model, cleaned)
            is_positive = p_pos >= THRESHOLD
            confidence = p_pos if is_positive else 1 - p_pos

            if is_positive:
                st.success("Positive review")
            else:
                st.error("Negative review")

            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{confidence:.1%}")
            col2.metric("Probability of positive", f"{p_pos:.1%}")
            st.progress(min(max(p_pos, 0.0), 1.0))

            if not is_calibrated:
                st.caption(
                    "The confidence is an approximate score from the model's decision value, "
                    "not a calibrated probability."
                )

            with st.expander("See the cleaned text the model received"):
                st.code(cleaned, language=None)
