# Sentiment Analysis: Text Classification App

> Text classification project with a Streamlit demo for predicting sentiment from text.

![Status](https://img.shields.io/badge/Status-Portfolio--Project-blue)

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Insights](#results-and-insights)
- [Screenshots](#screenshots)
- [Challenges and Solutions](#challenges-and-solutions)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [What I Learned](#what-i-learned)
- [Author](#author)
- [License](#license)

## 📖 Overview

Text classification project with a Streamlit demo for predicting sentiment from text. This README documents the intended workflow and the files available in the repository. Verify all outputs against the current code and data before using them as evidence.

## 🎯 Problem Statement

Manually reviewing large amounts of text is time-consuming; sentiment classification can help summarize language patterns when evaluated carefully.

## 🎯 Objectives

- Prepare text for model input.
- Compare implemented classification approaches.
- Expose inference through an app.

## ✨ Features

- Sentiment_Analysis.ipynb notebook.
- Saved pipeline and sentiment.py application code.
- Streamlit demo and requirements file.

## 📂 Dataset

P597 DATASET.xlsx is present. Verify the source, labels, language coverage, and license before describing the dataset.

- **Privacy and licensing:** Check source terms and avoid publishing personal, sensitive, or restricted data.
- **Preprocessing:** Follow the actual SQL/notebook/report implementation; do not assume cleaning steps or row counts that have not been verified.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and modelling |
| pandas/scikit-learn | Data preparation and classification |
| TF-IDF/NLP | Text representation where implemented |
| Streamlit | Interactive demo |

## 🔄 Project Workflow

1. Create an environment and install requirements.txt.
2. Inspect P597 DATASET.xlsx, labels, class balance, and data provenance.
3. Review Sentiment_Analysis.ipynb for preprocessing and feature extraction.
4. Split data before fitting the vectorizer or any resampling operation.
5. Train and compare only the models implemented in the notebook.
6. Evaluate with per-class precision, recall, F1, and a confusion matrix on held-out data.
7. Confirm pipeline.pkl and sentiment.py use the same preprocessing and model steps.
8. Run the app and test new examples, recording common errors.

## 🗂️ Repository Structure

```text
sentiment-analysis/
├── P597 DATASET.xlsx
├── Sentiment_Analysis.ipynb
├── sentiment.py
├── pipeline.pkl
├── requirements.txt
├── DEPLOYMENT.md
└── LIVE_APP.md
└── README.md
```

The list above highlights known project files and folders; inspect the repository for the complete current structure.

## ⚙️ Installation

### Prerequisites

- Install the tools listed in the Tech Stack that apply to this project.
- For Python projects, use a Python version compatible with the dependencies actually imported by the code.
- For Power BI projects, use Power BI Desktop.

### Setup

git clone https://github.com/sahil2003-ai/sentiment-analysis.git; cd sentiment-analysis; python -m venv .venv; activate the environment; pip install -r requirements.txt

Do not run generic commands that do not match the files in this repository. Confirm the data paths, database dialect, and package versions before execution.

## ▶️ Usage

Review LIVE_APP.md for the hosted link. For local use, inspect sentiment.py and run the actual Streamlit entry command documented in the repository.

## 📊 Results and Insights

### Key Findings

Add verified held-out metrics and an error analysis. No performance figures are asserted here.

### Evaluation

For analytics projects, reconcile key totals and document metric definitions. For ML projects, report the baseline, split strategy, relevant metrics, and held-out evaluation results. Do not include unverified scores or business impact claims.

## 🖼️ Screenshots

No image files were confirmed for this repository during this update. If you have screenshots, upload them into an `images/` folder, then embed each real path here with a short caption. Do not use placeholder image paths.

## 🧩 Challenges and Solutions

| Challenge | Solution / validation approach |
|---|---|
| Text preprocessing consistency | Package the fitted preprocessing and classifier together and verify inference matches training. |
| Class imbalance | If resampling is used, apply it to training data only. |

## ⚠️ Limitations

- Sentiment is context-dependent and may fail on sarcasm, domain-specific language, or unsupported languages.
- Dataset provenance and license should be documented.

## 🚀 Future Improvements

- [ ] Add an application screenshot under images/.
- [ ] Document exact local run commands.
- [ ] Publish a verified metrics table and error examples.

## 📚 What I Learned

- Text preprocessing and model evaluation.
- Packaging inference pipelines for an app.

## 👤 Author

**Sahil Gaikwad**

- GitHub: [sahil2003-ai](https://github.com/sahil2003-ai)
- LinkedIn: [Sahil Gaikwad](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)

## 📄 License

Check dataset terms and repository license before reuse.
