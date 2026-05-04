# Airline Review Sentiment Analysis

An NLP sentiment analysis pipeline on airline reviews demonstrating text preprocessing, classical machine learning, and transformer-based deep learning approaches. This project showcases end-to-end NLP workflow from raw text to sentiment predictions.

## Overview

This project covers:
- **Text Preprocessing**: Tokenisation, stemming, stopword removal
- **Feature Extraction**: TF-IDF vectorisation and embeddings
- **Classical ML**: Logistic Regression, Random Forest, and ensemble classifiers
- **Visualisation**: Word clouds, confusion matrices, sentiment distribution

## Dataset

**US Airline Twitter Sentiment Dataset**
- Tweet reviews of major US airlines
- ~14,000 reviews with sentiment labels (positive, neutral, negative)
- Timestamps and airline identifiers
- Multi-class sentiment classification task

## Tech Stack

- **Python 3.8+**
- **NLTK & TextBlob**: Text preprocessing and rule-based sentiment scoring
- **scikit-learn**: TF-IDF vectorisation and ML classifiers
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Visualisation
- **wordcloud**: Word frequency visualisation

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<username>/airline-sentiment-nlp.git
cd airline-sentiment-nlp
pip install -r requirements.txt
```

## Usage

Open and run the notebook:

```bash
jupyter notebook notebooks/airline_sentiment_analysis.ipynb
```

The notebook demonstrates an end-to-end NLP pipeline from raw tweet text to trained classifiers, with model comparison and evaluation.

## Results

The project evaluates multiple approaches:
- TF-IDF vectorisation with classical ML classifiers
- Cross-validation and test set performance
- Confusion matrices and classification reports
- Word cloud visualisations of sentiment-specific terms

## Model Comparison

- **Baseline (Naive Bayes + TF-IDF)**: Fast, interpretable
- **Advanced (Transformers)**: Higher accuracy, context-aware

## License

MIT License — feel free to use this project for learning and reference.
