import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download only once (stopwords + wordnet)
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Utility Functions
# ----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)                    # strip HTML
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)              # keep only letters/spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # simple regex tokenizer instead of word_tokenize
    tokens = re.findall(r'\b[a-z]{2,}\b', text)

    # stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {"the","and","is","in","it","of","to","a","that","this","for","on","with","as","are","was","my","i"}

    tokens = [w for w in tokens if w not in stop_words]

    # lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    except:
        tokens = tokens  # fallback if wordnet missing

    return ' '.join(tokens)


@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("cyberbullying_dataset.csv")
    df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
    le = LabelEncoder()
    df['Encoded'] = le.fit_transform(df['Is_Cyberbullying'])
    return df, le


@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Text'], df['Encoded'], test_size=0.2, stratify=df['Encoded'], random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer


def predict(text, model, vectorizer, df):
    cleaned = preprocess_text(text)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    prob = model.predict_proba(tfidf)[0][prediction]
    label = "Cyberbullying" if prediction == 1 else "Non-Cyberbullying"

    # Similarity
    all_text_tfidf = vectorizer.transform(df['Cleaned_Text'])
    similarity_scores = cosine_similarity(tfidf, all_text_tfidf)[0]
    top_indices = similarity_scores.argsort()[-4:][::-1]
    similar_df = df.iloc[top_indices]

    reason = similar_df.iloc[0]['Reason'] if 'Reason' in df.columns else "N/A"
    action = "Ban user" if "hate" in reason.lower() or "sexual" in reason.lower() else (
        "Restrict comments" if label == "Cyberbullying" else "No action")

    return {
        "label": label,
        "probability": prob,
        "similar_examples": similar_df['Text'].tolist(),
        "top_reason": reason,
        "action": action
    }


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config("🚫 Cyberbullying Detector", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.title("⚙️ Settings")
    dataset = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    theme = st.radio("Choose Theme", options=["Light", "Dark"])
    show_vis = st.checkbox("Show Data Visuals", value=True)

st.markdown(
    f"<style>body {{ background-color: {'#0e1117' if theme == 'Dark' else '#f8f9fa'}; }}</style>",
    unsafe_allow_html=True
)

st.markdown(f"""
    <h1 style="text-align: center;">🛡️ Cyberbullying Detection App</h1>
    <p style="text-align: center; font-size: 18px;">Detect harmful language and suggest moderation actions in real-time.</p>
""", unsafe_allow_html=True)

df, label_encoder = load_data(dataset)
model, vectorizer = train_model(df)

st.markdown("## ✍️ Enter a Message")
text = st.text_area("", placeholder="Type or paste any text to check for cyberbullying", height=150)

if st.button("🚨 Analyze"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        result = predict(text, model, vectorizer, df)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🔍 Prediction Result")
            st.markdown(f"""
                <div style="padding:1rem;border:2px solid {'red' if result['label']=="Cyberbullying" else 'green'}; border-radius:10px;">
                    <b>Classification:</b> {result['label']}  
                    <br><b>Confidence:</b> {result['probability']*100:.2f}%  
                    <br><b>Action:</b> {result['action']}  
                    <br><b>Reason:</b> {result['top_reason']}  
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("📊 Prediction Chart")

            # align probability with prediction
            if result['label'] == "Cyberbullying":
                cyber_prob = result['probability']
                non_cyber_prob = 1 - result['probability']
            else:
                non_cyber_prob = result['probability']
                cyber_prob = 1 - result['probability']

            fig = px.pie(
                names=["Cyberbullying", "Non-Cyberbullying"],
                values=[cyber_prob, non_cyber_prob],
                color=["Cyberbullying", "Non-Cyberbullying"],
                color_discrete_map={"Cyberbullying": "red", "Non-Cyberbullying": "green"},
                hole=0.4
            )
            st.plotly_chart(fig)

            st.markdown("---")
            st.markdown("### 🔁 Most Similar Examples")
            for i, example in enumerate(result['similar_examples']):
                st.markdown(f"**{i+1}.** _{example}_")

# ----------------------------
# Data Visualization
# ----------------------------
if show_vis:
    st.markdown("---")
    st.markdown("## 📈 Dataset Overview")
    class_dist = df['Is_Cyberbullying'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x=class_dist.index, y=class_dist.values, palette="Set2", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    if 'Reason' in df.columns:
        st.markdown("### 📌 Top 5 Reasons")
        top_reasons = df['Reason'].value_counts().head(5)
        st.bar_chart(top_reasons)

st.markdown("----")
st.markdown("<small>© 2025 CyberShield AI</small>", unsafe_allow_html=True)
