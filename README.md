# Sentiment Analysis: Text Classification App

A Python NLP project that processes text, assigns sentiment labels, trains classification models, and provides a Streamlit interface for new-text predictions.

![Status](https://img.shields.io/badge/Status-Documented-yellow)

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results and Insights](#-results-and-insights)
- [Screenshots](#-screenshots)
- [Challenges and Solutions](#-challenges-and-solutions)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [What I Learned](#-what-i-learned)
- [Author](#-author)
- [License](#-license)

## 📖 Overview

A Python NLP project that processes text, assigns sentiment labels, trains classification models, and provides a Streamlit interface for new-text predictions.

## 🎯 Problem Statement

The project addresses an analytical or predictive question using the data and implementation files in the repository. Results are meaningful only when dataset definitions, preprocessing, and evaluation design are transparent.

## 🎯 Objectives

- Inspect and prepare the available data.
- Implement the analysis or model workflow documented in the project.
- Report verifiable metrics and distinguish saved outputs from reproduced results.
- Make the project easier to reproduce and review.

## ✨ Features

- Load and inspect the spreadsheet.
- Clean and normalize text, including stopword removal and lemmatization where configured.
- Assign sentiment using TextBlob polarity thresholds in the notebook.
- Convert text into features with a TF-IDF workflow and train candidate classifiers.
- Select a model based on validation/test scores as implemented in the notebook.
- Save a pipeline and use it for app inference.

## 📂 Dataset

Dataset details and numerical metrics are documented only where the existing repository files provide evidence. See the source files and notebook outputs for the precise schema and data provenance.

## 🛠️ Tech Stack

See the imports, dependency files, SQL scripts, and application code in the repository. Only technologies directly evidenced in those files should be listed as used.

## 🔄 Project Workflow

- Load and inspect the spreadsheet.
- Clean and normalize text, including stopword removal and lemmatization where configured.
- Assign sentiment using TextBlob polarity thresholds in the notebook.
- Convert text into features with a TF-IDF workflow and train candidate classifiers.
- Select a model based on validation/test scores as implemented in the notebook.
- Save a pipeline and use it for app inference.

## 🗂️ Repository Structure

| File | Purpose |
|---|---|
| `P597 DATASET.xlsx` | Text dataset |
| `Sentiment_Analysis.ipynb` | Text processing and model experiments |
| `sentiment.py` | Application / preprocessing logic |
| `pipeline.pkl` | Saved text-classification pipeline |
| `requirements.txt` | Dependencies |
| `DEPLOYMENT.md`, `LIVE_APP.md` | Deployment documentation |

## ⚙️ Installation

### Prerequisites\n\nUse the runtime and dependencies required by the actual source files. Check the repository for a requirements or environment file before installing packages.\n\n### Setup\n\n\`\`\`bash\ngit clone https://github.com/sahil2003-ai/${repo}.git\ncd ${repo}\n\`\`\`

## ▶️ Usage

Open the notebook or application entry point actually present in this repository and follow its input-path and dependency instructions. Do not assume an entry script or deployment URL that is not documented.

## 📊 Results and Insights

No numerical results were independently confirmed in the current README content. Add metrics only after reproducing the calculations from the repository data or saved outputs.

## 🖼️ Screenshots

Add only genuine charts or application screenshots that exist in the repository. No screenshot path is asserted here unless documented in the source files.

## 🧩 Challenges and Solutions

| Challenge | Solution |
|---|---|
| Saved notebook results can become stale | Rerun cells from top to bottom and verify outputs. |
| Model performance can be overstated | Use a valid holdout or cross-validation and report the split strategy. |
| Reproduction depends on paths and package versions | Document dependencies, input paths, and data provenance. |

## ⚠️ Limitations

### Limitations

- Numerical model results and dataset dimensions were not independently confirmed here.
- Rule-generated labels can create circular evaluation if the model is tested only against those labels.
- Text data may contain copyrighted or private content; verify redistribution permissions.

## 🚀 Future Improvements

### Validation priorities

- Record the exact dataset dimensions, missing values, label counts, language, provenance, and license.
- Confirm the train/test split occurs before fitting vectorizers and any transformations with learned state.
- Save per-class precision, recall, F1, confusion matrix, and validation strategy.
- Evaluate on independently annotated text if the goal is real-world sentiment performance.
- Check that `pipeline.pkl` and the application apply matching text preprocessing.

## 📚 What I Learned

- Building a documented and reproducible analysis workflow.
- Choosing metrics appropriate to the task and dataset.
- Explaining assumptions and limitations alongside results.

## 👤 Author

**Sahil Gaikwad**

- GitHub: [sahil2003-ai](https://github.com/sahil2003-ai)
- LinkedIn: [Sahil Gaikwad](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)

## 📄 License

No verified license statement was found in the README. Check the repository's LICENSE file and the source dataset's reuse terms before redistribution.