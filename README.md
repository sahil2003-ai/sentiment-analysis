# Sentiment Analysis: Text Classification App

A text-classification project that processes written reviews and predicts sentiment labels. The repository documents a Streamlit demo and experiments with multiple traditional machine-learning models.

## Live Demo
[Open the Streamlit app](https://sentiment-analysis-s3chle3ar9fjixe4f6mzyk.streamlit.app/)

## Project Goals
- Clean and normalize text for analysis.
- Convert text into numerical features.
- Compare classifiers for positive, negative, and neutral sentiment.
- Present predictions through an interactive interface.

## Pipeline Documented
- Text cleanup, tokenization, and lemmatization
- TF-IDF feature extraction
- Class-imbalance handling with SMOTE
- Experiments with Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, KNN, XGBoost, and MLP
- Hyperparameter search using GridSearchCV
- Streamlit-based inference interface

Check the source code to confirm which steps and models are included in the current runnable pipeline.

## Tools
Python, scikit-learn, TF-IDF, imbalanced-learn/SMOTE where used, NLP preprocessing libraries, and Streamlit.

## Reproducibility and Evaluation
1. Document the dataset source, label definitions, and language coverage.
2. Split data before fitting vectorizers or applying resampling.
3. Apply SMOTE only to training data.
4. Report per-class precision, recall, F1-score, confusion matrix, and a baseline.
5. Save the fitted preprocessing and model pipeline together.
6. Test the app with examples not used during training.

## Limitations
Sentiment is context-dependent. Translation and language detection can introduce errors, and model performance may vary across topics, languages, and writing styles.

## Recommended Improvements
Add exact install/run instructions, dataset licensing, verified test metrics, screenshots, and a short error analysis. Avoid calling the approach state-of-the-art unless supported by comparative evidence.