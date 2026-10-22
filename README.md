# IMDB Movie Review Sentiment Analysis

Classifying IMDB movie reviews as **positive** or **negative** with classical NLP and machine learning. The project covers the full workflow: data quality checks, EDA, text cleaning, normalization comparison, a vectorizer and model comparison, hyperparameter tuning, threshold and calibration analysis, and a Streamlit app for live predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Results

| Item | Value |
|------|-------|
| Final model | TF-IDF (1-2 grams) + Linear SVC |
| Best parameters | `C=0.5`, `max_df=0.9`, `min_df=2` |
| Cross-validated F1 (train + validation, 5-fold) | 0.9095 |
| Test accuracy | about 0.91 |
| Best decision threshold (out-of-fold F1) | 0.475 |
| Brier score of calibrated probabilities | 0.0659 |

**Test-set performance (evaluated once, after all tuning):**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Positive | 0.90 | 0.92 | 0.91 |
| Negative | 0.92 | 0.90 | 0.91 |

## Project Workflow

| # | Stage | What was done |
|---|-------|---------------|
| 1 | Data loading and quality check | Shape, types, missing values, duplicate removal |
| 2 | EDA | Class balance, review length, noise statistics, frequent words, word clouds |
| 3 | Text cleaning | HTML and bracket removal, special characters, lowercasing, stopword removal |
| 4 | Normalization comparison | Basic vs stemming vs lemmatization (Basic selected) |
| 5 | Split | Stratified 60 / 20 / 20 train, validation and test |
| 6 | Vectorizer + model comparison | BoW, TF-IDF, and both with 1-2 grams, with Logistic Regression, Naive Bayes and Linear SVC (12 combinations) |
| 7 | Hyperparameter tuning | `GridSearchCV`, 5-fold stratified CV on the top 3 combinations |
| 8 | Final evaluation | One-time evaluation on the test set |
| 9 | Interpretation | Most influential words and phrases per class |
| 10 | Threshold and calibration | Precision/recall trade-off and calibration curve |

## Key Decisions

- **Negation words are kept.** `not`, `no`, `nor` and `never` are removed from the NLTK stopword list, because dropping them would turn "not good" into "good".
- **HTML is stripped with a space separator**, so `<br />` does not merge neighbouring words.
- **No data leakage.** Cleaning and normalization are stateless and run before the split. Anything that learns from data (vectorizer, model) is fitted on training data only, using `Pipeline`.
- **Basic normalization won.** Stemming and lemmatization did not improve validation scores.
- **Bigrams helped.** The top combinations all use 1-2 grams, which capture phrases such as "not good".
- **Honest evaluation.** The test set is used once. Model selection and tuning use validation data and cross-validation only.

## Repository Structure

```
imdb-sentiment-analysis/
├── notebooks/
│   └── imdb_nlp.ipynb            # Full analysis and modeling notebook
├── data/
│   └── .gitkeep                  # Place imdb_dataset.csv here (not tracked)
├── models/
│   └── final_sentiment_pipeline.joblib   # Fitted vectorizer + model
├── images/                       # Plots used in this README
├── app/                          # Streamlit app
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the dataset

The dataset is not stored in this repository because of its size. Download **IMDB Dataset of 50K Movie Reviews** from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), rename the file to `imdb_dataset.csv`, and place it in the `data/` folder.

### 4. Download the NLTK data (first run only)

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### 5. Run the notebook

```bash
jupyter notebook notebooks/imdb_nlp.ipynb
```

### 6. Run the Streamlit app

```bash
streamlit run app/app.py
```

## Methodology Notes

**Vectorizers compared:** BoW (`CountVectorizer`), TF-IDF (`TfidfVectorizer`), and both with `ngram_range=(1, 2)`.

**Models compared:** Logistic Regression, Multinomial Naive Bayes, Linear SVC.

**Validation ranking (top 3 by F1-score):**

| Rank | Vectorizer | Model | F1-score |
|------|-----------|-------|----------|
| 1 | TF-IDF + n-grams | Linear SVC | 0.9073 |
| 2 | BoW + n-grams | Logistic Regression | 0.9001 |
| 3 | TF-IDF + n-grams | Logistic Regression | 0.8998 |

**Tuning grid:** `min_df` in {2, 5}, `max_df` in {0.9, 1.0}, plus `C` for Logistic Regression and Linear SVC, or `alpha` for Naive Bayes.

**Probabilities:** Linear SVC has no `predict_proba`, so the threshold and calibration analysis uses `CalibratedClassifierCV` (sigmoid, 5-fold) around it.

## Limitations

- Bag-of-words models ignore word order beyond 2-grams and miss context.
- Sarcasm and mixed reviews ("great acting, terrible plot") are often misclassified.
- The model was trained on one dataset of movie reviews and may not transfer to other domains such as product reviews or tweets.

## Future Work

- Error analysis of misclassified reviews
- Word embeddings (Word2Vec, GloVe) or a fine-tuned transformer such as DistilBERT for comparison
- Wrapping the cleaning step into the pipeline with `FunctionTransformer`

## Tech Stack

Python, pandas, NumPy, scikit-learn, NLTK, BeautifulSoup, matplotlib, seaborn, WordCloud, joblib, Streamlit.

## Author

**Güldadaş**, Computer Engineering student focused on data science and machine learning.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
