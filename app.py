import streamlit as st
import pickle
import numpy as np

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis App")

review = st.text_area("Enter a review to analyze", height=150)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter some text!")
    else:
        vec = tfidf.transform([review])
        pred = model.predict(vec)[0]

        # Adjust mapping below to match your labels
        # Example mappings:
        # Binary -> {0: "Negative", 1: "Positive"}
        # Multi-class -> {0: "Negative", 1: "Neutral", 2: "Positive"}
        mapping = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

        # If your labels are strings, you can do:
        # mapping = {"negative":"Negative 😞", "neutral":"Neutral 😐", "positive":"Positive 😊"}

        label = mapping.get(pred, str(pred))
        if "Positive" in label:
            st.success("Sentiment: " + label)
        elif "Neutral" in label:
            st.info("Sentiment: " + label)
        else:
            st.error("Sentiment: " + label)



