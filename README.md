# Sentiment Analysis — Multilingual Text Classification

An end-to-end NLP project that processes review titles and bodies, normalizes multilingual text, creates sentiment labels, trains and compares text-classification models, and serves predictions through a Streamlit application.

<p align="left">
  <img src="https://img.shields.io/badge/Python-NLP-3776AB?logo=python&logoColor=white" alt="Python NLP" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20TextBlob-6A5ACD" alt="TF-IDF and TextBlob" />
  <img src="https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-F7931E" alt="Machine learning" />
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

## Project overview

This project explores how natural-language reviews can be cleaned, represented as numerical features and classified into **positive, neutral or negative** sentiment categories. It combines language detection and translation, text normalization, exploratory analysis, candidate-model comparison and a saved TF-IDF/classifier pipeline for interactive inference.

The repository includes a review dataset in Excel format, a Jupyter notebook, application code, a saved pipeline and deployment documentation.

## Live demo

**[Open the Sentiment Analysis App](https://sentiment-analysis-s3chle3ar9fjixe4f6mzyk.streamlit.app/)**

The Streamlit interface accepts a review title, a review body, or both, and returns a predicted sentiment label.

## Repository contents

| File | Purpose |
|---|---|
| [`P597 DATASET.xlsx`](P597%20DATASET.xlsx) | Source review dataset |
| [`Sentiment_Analysis.ipynb`](Sentiment_Analysis.ipynb) | Text preprocessing, sentiment labeling, visualization and model experiments |
| [`sentiment.py`](sentiment.py) | Streamlit application entry point |
| [`pipeline.pkl`](pipeline.pkl) | Saved TF-IDF and classifier pipeline used for app predictions |
| [`requirements.txt`](requirements.txt) | Python package requirements |
| [`LIVE_APP.md`](LIVE_APP.md) | Live app details and usage instructions |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Deployment notes |

## Dataset and profile

The notebook loads review data from `P597 DATASET.xlsx` using pandas. Its primary fields include **`title`** and **`body`**, which are combined into a review text field for analysis. The full row count, full schema, class frequencies and missing-value totals should be regenerated from the workbook before publishing exact dataset statistics; numeric dataset size is not asserted here without a confirmed current profile.

### Label construction

The notebook uses TextBlob polarity to generate three labels:

| TextBlob polarity | Generated label |
|---|---|
| Greater than 0.1 | Positive |
| Less than -0.1 | Negative |
| Between -0.1 and 0.1, inclusive | Neutral |

These labels are **rule-generated sentiment targets**, not independently verified human annotations. As a result, the classifier is learning to reproduce the labeling approach used in the notebook. This is an important limitation when interpreting model evaluation.

## NLP preprocessing workflow

The notebook's `TextPreprocessor` class provides configurable preprocessing steps:

1. **Language detection** with `langdetect`.
2. **Translation workflow** using `deep-translator` when a detected language differs from the configured target language.
3. **Text normalization** — lowercase text, remove emojis, URLs, HTML-like tags, punctuation, digits and repeated whitespace.
4. **Tokenization** using spaCy.
5. **Stopword removal** using NLTK.
6. **Lemmatization** using NLTK's WordNet lemmatizer.
7. **Text combination** — the translated title and review body are concatenated.
8. **Sentiment labeling** — TextBlob polarity thresholds create positive, neutral or negative labels.

The app loads the saved classification pipeline directly. Its inference path should be checked against the notebook's full preprocessing chain, because the app currently passes the entered title/body text to the pipeline without calling the notebook's language-detection, translation, cleaning and lemmatization functions first. Matching training and inference preprocessing is necessary for reliable results.

## Feature engineering and model experiments

The notebook uses **TF-IDF** to represent processed text numerically, with a vocabulary capped at 1,000 features and unigram/bigram features.

The experiment defines and compares eight candidate classifiers:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- Multi-layer Perceptron (MLP)

It defines hyperparameter grids and uses stratified five-fold cross-validation with accuracy as the selection score. The notebook also applies SMOTE to TF-IDF features in the training workflow.

### Model selection and saved pipeline

The notebook output available in the repository shows the fitted pipeline with a **TF-IDF vectorizer and MLP classifier**, and serializes it to `pipeline.pkl`. The live app loads this file for predictions.

A numeric test accuracy, macro-F1 or per-class metrics are not stated here because the exact saved evaluation table could not be confirmed in the current review. The separate live-app notes mention an accuracy claim, but it should be verified against regenerated results and a fully reproducible evaluation before being treated as an established metric.

## Results and evaluation

### Confirmed artifacts and workflow

| Item | Documented status |
|---|---|
| Sentiment classes | Positive, neutral, negative |
| Text representation | TF-IDF with unigrams and bigrams |
| TF-IDF vocabulary cap | 1,000 features |
| Candidate classifiers defined | 8 |
| Cross-validation strategy | Stratified 5-fold CV, shuffled with random state 42 |
| Saved inference artifact | `pipeline.pkl` |
| Interactive interface | Streamlit app |
| Example prediction in notebook | Example positive review is classified as positive |

### Metrics to report after a clean rerun

For an informative and honest model report, regenerate a results table with:

- Accuracy
- Precision, recall and F1-score for each class
- Macro-F1 and weighted-F1
- Confusion matrix
- Class distribution before and after any resampling
- Cross-validation score and independent test score
- Baseline comparison (for example, majority-class prediction)
- The data split, random seed and preprocessing steps used

**Important evaluation note:** the notebook creates labels from TextBlob polarity, applies SMOTE before the grid search, and then compares against a held-out TF-IDF test matrix. Review the resampling/vectorization sequence to ensure the test split remains untouched and that no data leakage affects evaluation. For stronger evidence, fit vectorization and resampling within an appropriate training-only pipeline and evaluate against a separate untouched test set.

## Visual analysis in the notebook

The notebook generates several visuals during execution:

- Sentiment class count chart
- Sentiment proportion chart
- Separate word clouds for positive, negative and neutral reviews
- Confusion matrix for the classifier

The repository currently contains no separate screenshot/image files. These visuals are generated inside the notebook and are not available as standalone files for embedding in the README. To display them on the repository landing page, export genuine plots from a fresh notebook run into an `images/` folder and reference the resulting paths here. No placeholder or unrelated screenshots have been used.

## Streamlit application

The application in [`sentiment.py`](sentiment.py) provides:

- Review-title input
- Review-body input
- Support for entering either field independently or both together
- Prediction into positive, neutral or negative sentiment
- Color and emoji indicators for the predicted class
- Cached loading of the saved pipeline for repeat predictions

The app is intended for short text-review examples. Predictions should be treated as model outputs, not as ground truth.

## Technology stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data handling | pandas, NumPy |
| Language detection | langdetect |
| Translation | deep-translator |
| Text processing | regex, string utilities, emoji |
| NLP | spaCy, NLTK, TextBlob |
| Text representation | scikit-learn TF-IDF |
| Modeling | scikit-learn classifiers, XGBoost, imbalanced-learn/SMOTE |
| Visualization | Matplotlib, Seaborn, WordCloud |
| App / serialization | Streamlit, joblib |

Install dependencies from [`requirements.txt`](requirements.txt). The notebook downloads NLTK resources and loads spaCy's `en_core_web_sm` model; these need to be available in the runtime environment as well.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/sahil2003-ai/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The notebook may also need NLTK resources such as `stopwords` and `wordnet`; download them during notebook setup if missing.

### 4. Run the Streamlit app

```bash
streamlit run sentiment.py
```

Run from the repository root so the application can find `pipeline.pkl`.

### 5. Re-run analysis

Open [`Sentiment_Analysis.ipynb`](Sentiment_Analysis.ipynb), confirm the input workbook path, review the labeling/preprocessing steps, and execute cells in order. Regenerate class counts and test metrics before publishing any new performance claims.

## Limitations and next steps

- **Dataset provenance:** document the workbook's original source, source date, row count and license.
- **Rule-generated labels:** TextBlob-generated labels can be noisy, especially for sarcasm, mixed sentiment, domain terminology and translated text.
- **Translation reliability:** online translation can change meaning and depends on an external service; document error handling and language coverage.
- **Preprocessing mismatch:** the app's current inference path does not run the same custom preprocessing chain shown in the notebook. Package preprocessing and vectorization into a single tested inference pipeline.
- **Evaluation integrity:** confirm no resampling or data leakage affects test metrics; report class-wise and macro metrics, not accuracy alone.
- **Reproducibility:** pin versions for packages and language resources, and run a clean installation test.
- **Visual evidence:** export sentiment charts, word clouds and the confusion matrix to `images/` after a successful notebook execution.
- **Privacy:** do not upload private customer reviews or personally identifying information without an appropriate basis and safeguards.

## Skills demonstrated

Natural-language processing · text cleaning · language detection · translation workflow · tokenization · stopword removal · lemmatization · polarity-based sentiment labeling · TF-IDF · model comparison · cross-validation · class resampling · confusion-matrix analysis · model serialization · Streamlit

## Author

**Sahil Gaikwad**  
[GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
