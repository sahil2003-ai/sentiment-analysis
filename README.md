# Sentiment Analysis: Text Classification App

A text-classification project with a Streamlit demo for predicting sentiment from text.

## Live demo
[Try the Streamlit app](https://sentiment-analysis-s3chle3ar9fjixe4f6mzyk.streamlit.app/)

## Step-by-step workflow

1. Create a Python environment and install dependencies from `requirements.txt`.
2. Inspect `P597 DATASET.xlsx` and document its source, language coverage, target labels, and class balance.
3. Open `Sentiment_Analysis.ipynb` and inspect the cleaning and preprocessing pipeline.
4. Split the data into training and test sets before fitting the vectorizer or resampling.
5. Transform text with the documented preprocessing and TF-IDF steps.
6. Train and compare the models actually implemented in the notebook.
7. Apply SMOTE only to the training data if used; never resample the test set.
8. Evaluate on held-out data using per-class precision, recall, F1-score, and a confusion matrix.
9. Confirm `pipeline.pkl` contains the same preprocessing and model steps used by `sentiment.py`.
10. Test the app with new text and record common errors.

## Tools
Python, pandas, scikit-learn, NLP preprocessing, TF-IDF, Streamlit.

## Limitations
Language detection or translation can introduce errors. Sentiment depends on context, sarcasm, domain, and writing style.

## Screenshots
No standalone screenshot images were found in the repository root. Add an app screenshot under `images/` and embed it here.

## Next improvements
Document exact model metrics, dataset license, installation/run commands, and examples of incorrect predictions.