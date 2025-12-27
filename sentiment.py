import streamlit as st
import joblib

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #dfe9f3 0%, #ffffff 100%); }
    .title { font-size: 64px; font-weight: 900; text-align: center; color: #3A7BD5; margin-bottom: 0px; }
    .result-text { font-size: 26px; font-weight: 700; margin-top: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return joblib.load("pipeline.pkl")  # Use your saved pipeline filename

pipeline = load_pipeline()

st.markdown('<div class="title">🔍 Sentiment Analysis</div>', unsafe_allow_html=True)

with st.container():
    user_title = st.text_input("Enter Review Title:")
    user_body = st.text_area("Enter Review Body:", height=100)

def predict_sentiment(text):
    prediction = pipeline.predict([text])[0]
    if prediction == 2:
        label = "Positive 😊"
        color = "green"
    elif prediction == 1:
        label = "Neutral 😐"
        color = "gray"
    else:
        label = "Negative 😠"
        color = "red"
    return label, color

if st.button("Analyze Sentiment"):
    # Handle cases where one or both fields are empty
    if user_title.strip() == "" and user_body.strip() == "":
        st.warning("⚠️ Please enter either a review title or body (or both).")
    else:
        if user_title.strip() != "" and user_body.strip() != "":
            review_text = f"{user_title.strip()} {user_body.strip()}"
        else:
            # If one column is missing, use only the one that's filled
            review_text = user_title.strip() if user_body.strip() == "" else user_body.strip()

        sentiment, color = predict_sentiment(review_text)
        st.markdown(
            f'<div class="result-text" style="color:{color}">Sentiment: {sentiment}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("You may enter either the Title, Body, or both for sentiment analysis. The app will work even if you fill only one field. For more features or batch analysis, request expansion!")
