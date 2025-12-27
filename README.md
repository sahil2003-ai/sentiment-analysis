# Sentiment Analysis

A comprehensive machine learning project for multi-class sentiment classification using deep text preprocessing and multiple ML algorithms. This project analyzes text reviews and classifies them into **Positive**, **Negative**, or **Neutral** sentiments with state-of-the-art preprocessing techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project implements a complete sentiment analysis pipeline with the following highlights:

- **Multi-language Support**: Automatic language detection and translation to English
- **Advanced Text Preprocessing**: Emoji removal, URL cleaning, tokenization, lemmatization
- **8 ML Models**: Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, KNN, XGBoost, MLP
- **Class Balancing**: SMOTE algorithm for handling imbalanced datasets
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Interactive Web App**: Streamlit-based user interface for real-time predictions

## ✨ Features

### Text Processing Pipeline
- **Language Detection & Translation**: Identifies input language and translates to English for uniform processing
- **Emoji Handling**: Removes emojis while preserving sentiment information
- **URL & HTML Cleaning**: Strips URLs and HTML tags
- **Punctuation & Number Removal**: Cleans text for better feature extraction
- **Tokenization**: Breaks text into individual words using spaCy
- **Stop Words Removal**: Filters common English words
- **Lemmatization**: Converts words to their base form

### Machine Learning
- **Feature Extraction**: TF-IDF vectorization with bigrams (max 1000 features)
- **Class Balancing**: SMOTE for imbalanced sentiment distribution
- **Model Training**: 8 different algorithms with cross-validation
- **Hyperparameter Optimization**: GridSearchCV with 5-fold stratified cross-validation
- **Performance Evaluation**: Accuracy, precision, recall, and confusion matrix analysis

### Visualization
- Sentiment distribution bar charts and pie charts
- Word clouds for each sentiment class
- Confusion matrices for model evaluation

### Web Application
- Clean, modern Streamlit interface
- Support for both title and body review input
- Flexible input (accept either title, body, or both)
- Real-time sentiment prediction with emoji indicators
- Responsive design with gradient background

## 📁 Project Structure

```
sentiment-analysis/
├── Sentiment_Analysis.ipynb      # Main Jupyter notebook with full pipeline
├── sentiment.py                  # Streamlit web application
├── pipeline.pkl                  # Trained ML pipeline (serialized)
├── P597 DATASET.xlsx             # Dataset with reviews and sentiments
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/sahil2003-ai/sentiment-analysis.git
cd sentiment-analysis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install pandas numpy matplotlib seaborn nltk spacy scikit-learn xgboost imbalanced-learn textblob wordcloud langdetect deep-translator streamlit joblib
```

### Step 3: Download Required NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## 📖 Usage

### Option 1: Interactive Web Application

Run the Streamlit app for an interactive interface:

```bash
streamlit run sentiment.py
```

Then:
1. Open your browser to `http://localhost:8501`
2. Enter a review title and/or body
3. Click "Analyze Sentiment" to get predictions
4. View the sentiment classification (Positive 😊, Neutral 😐, or Negative 😠)

### Option 2: Jupyter Notebook

Run the complete analysis in Jupyter:

```bash
jupyter notebook Sentiment_Analysis.ipynb
```

This notebook contains:
- Data loading and exploration
- Text preprocessing implementation
- Model training and evaluation
- Visualization of results
- Pipeline creation and serialization

## 📊 Dataset

**File**: `P597 DATASET.xlsx`

**Columns**:
- `title`: Review title/heading
- `body`: Review content/description
- `sentiment`: Target label (positive, negative, neutral)

**Data Processing**:
- Language detection for title and body
- Automatic translation to English for non-English text
- Combined title and body into single text field
- Full text preprocessing as described above

## 🤖 Model Architecture

### Pipeline Components

1. **Text Vectorization**: TF-IDF with unigrams and bigrams
   - Max features: 1000
   - N-gram range: (1, 2)

2. **Best Model**: MLP Classifier (Neural Network)
   - Architecture: [100, 50] hidden layers
   - Activation: tanh
   - Solver: lbfgs
   - Max iterations: 300+

3. **Class Mapping**:
   - 0 = Negative
   - 1 = Neutral  
   - 2 = Positive

### Models Evaluated

| Model | CV Accuracy | Test Accuracy | Best For |
|-------|------------|---------------|----------|
| Logistic Regression | High | High | Baseline, Fast |
| Naive Bayes | Good | Good | Probabilistic |
| SVM | High | High | Margin Maximization |
| Decision Tree | Medium | Medium | Interpretability |
| Random Forest | High | High | Ensemble |
| KNN | Medium | Medium | Non-parametric |
| XGBoost | High | High | Gradient Boosting |
| MLP Neural Network | **Best** | **Best** | Non-linear Patterns |

## 📈 Results

### Key Metrics
- **Best Model**: MLP Neural Network
- **Train Accuracy**: ~95%+
- **Test Accuracy**: ~90%+
- **Cross-validation Score**: Consistent across 5 folds

### Visualizations Generated
- Sentiment distribution across dataset
- Word clouds for each sentiment class
- Confusion matrix for model predictions
- Classification reports with precision/recall per class

## 🛠️ Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and metrics
- **xgboost**: Gradient boosting framework
- **imbalanced-learn**: SMOTE for class balancing

### NLP Libraries
- **nltk**: Natural language toolkit (stopwords, lemmatization)
- **spaCy**: Advanced NLP tasks (tokenization, POS tagging)
- **textblob**: Sentiment analysis baseline
- **langdetect**: Language detection
- **deep-translator**: Text translation

### Visualization & Interface
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **wordcloud**: Word cloud generation
- **streamlit**: Web application framework

### Serialization
- **joblib**: Model and pipeline persistence

## 🔄 Workflow

```
Raw Input Text
      ↓
Language Detection & Translation
      ↓
Text Cleaning (emoji, URL, punctuation removal)
      ↓
Tokenization
      ↓
Stop Words Removal
      ↓
Lemmatization
      ↓
TF-IDF Vectorization
      ↓
ML Model Prediction
      ↓
Sentiment Output (Positive/Neutral/Negative)
```

## 📝 Example Usage

### In Python

```python
import joblib

# Load the pre-trained pipeline
pipeline = joblib.load('pipeline.pkl')

# Make predictions
text = "This product is amazing! I love it."
prediction = pipeline.predict([text])[0]

# Map prediction to label
sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Sentiment: {sentiment_map[prediction]}")
# Output: Sentiment: positive
```

## 🎓 Learning Outcomes

This project demonstrates:
- Complete NLP pipeline development
- Multi-language text processing
- Ensemble and deep learning methods
- Hyperparameter optimization techniques
- Web application deployment
- Model serialization and production deployment
- Data visualization and analysis

## 📝 Notes

- The dataset contains reviews in multiple languages which are automatically detected and translated
- SMOTE balancing is applied only during training to prevent data leakage
- The pipeline uses TF-IDF with bigrams to capture contextual information
- All preprocessing steps are integrated into the pipeline for seamless prediction

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created by [sahil2003-ai](https://github.com/sahil2003-ai)

## ⭐ If you found this helpful, please consider giving it a star!

---

**Last Updated**: December 2025
**Status**: Complete and Production-Ready
