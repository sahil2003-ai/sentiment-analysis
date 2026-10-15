# Sentiment Analysis: Text Classification App

A Python NLP project that processes text, assigns sentiment labels, trains classification models, and provides a Streamlit interface for new-text predictions.

![Python](https://img.shields.io/badge/Python-NLP-blue) ![App](https://img.shields.io/badge/App-Streamlit-FF4B4B)

## Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [NLP Workflow](#nlp-workflow)
- [Results and Metrics](#results-and-metrics)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Run the Project](#run-the-project)
- [Limitations and Next Steps](#limitations-and-next-steps)

## Overview
This project combines text preprocessing, sentiment labeling, TF-IDF features, candidate classifiers, and a saved pipeline for inference. The notebook uses TextBlob polarity thresholds to generate sentiment labels and compares classification models as implemented in the experiment.

## Dataset
The repository includes `P597 DATASET.xlsx`. The precise row count, column schema, source provenance, and reuse terms should be recorded from a fresh data profile before publishing dataset statistics.

## NLP Workflow
1. Load and inspect the spreadsheet.
2. Clean and normalize text; apply stopword removal and lemmatization where configured.
3. Assign sentiment labels using the notebook's TextBlob polarity thresholds.
4. Convert text to numeric features using TF-IDF.
5. Train and compare the candidate classifiers using the notebook's split strategy.
6. Save the text-classification pipeline and use it for app inference.

## Results and Metrics
The repository includes the notebook, application logic, and saved `pipeline.pkl`. No numeric accuracy/F1 result is quoted here because the evaluation output could not be independently confirmed from the current repository summary. After rerunning, report the train/test strategy, class distribution, accuracy, precision, recall, and macro-F1; compare with a simple baseline.

## Technology Stack
- Python
- pandas and spreadsheet-reading tools used by the notebook
- TextBlob for polarity-based labels
- TF-IDF text features and scikit-learn-style classifiers
- Streamlit for interactive inference
- joblib/pickle model pipeline

## Repository Structure
```text
sentiment-analysis/
├── P597 DATASET.xlsx
├── Sentiment_Analysis.ipynb
├── sentiment.py
├── pipeline.pkl
├── requirements.txt
├── DEPLOYMENT.md
├── LIVE_APP.md
└── README.md
```

## Run the Project
1. Review `requirements.txt`, `DEPLOYMENT.md`, and `LIVE_APP.md`.
2. Create a Python environment and install dependencies.
3. Open `Sentiment_Analysis.ipynb` and run cells in order, updating the dataset path if needed.
4. Follow the documented app entry point and verify that the pipeline loads successfully.
5. Test the same text preprocessing during training and inference.

## Limitations and Next Steps
- Document dataset provenance, label definitions, and licensing.
- Check for class imbalance and label noise introduced by polarity thresholds.
- Report per-class metrics and a confusion matrix, not accuracy alone.
- Ensure app preprocessing matches training preprocessing.
- Add genuine app screenshots and verified evaluation outputs.

## Skills Demonstrated
Text preprocessing, sentiment labeling, TF-IDF feature extraction, classification experiments, model persistence, and Streamlit deployment workflow.

## Author
**Sahil Gaikwad** · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
