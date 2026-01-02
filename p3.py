pip install streamlit feedparser pandas scikit-learn wordcloud matplotlib
import streamlit as st
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ---------- Helper Function ----------
def generate_wordcloud(topic, min_words=500, max_words=5000):
    url = f"https://www.reddit.com/search.rss?q={topic}&sort=hot"
    feed = feedparser.parse(url)

    titles = [entry.title for entry in feed.entries]

    if not titles:
        return None, "No data found"

    text = " ".join(titles)
    words_count = len(text.split())

    if words_count < min_words:
        return None, f"Only {words_count} words found. Try a broader topic."

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(titles)

    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    tfidf_dict = dict(zip(words, scores))

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white"
    ).generate_from_frequencies(tfidf_dict)

    return wc, f"Word count: {words_count}"


# ---------- UI ----------
st.set_page_config(page_title="AI Word Cloud Explorer", layout="wide")

st.title("🧠 Word Cloud Generator (TF-IDF Based)")
st.write("Explore trending topics using Reddit data")

tab1, tab2, tab3 = st.tabs(["🔥 Trending Tech", "🤖 AI Topics", "📊 Data Science"])


# ---------- TAB 1 ----------
with tab1:
    st.subheader("🔥 Trending Tech")
    topic1 = st.text_input("Enter topic", "artificial intelligence", key="t1")

    if st.button("Generate Word Cloud", key="b1"):
        wc, msg = generate_wordcloud(topic1)
        st.info(msg)

        if wc:
            st.image(wc.to_array())


# ---------- TAB 2 ----------
with tab2:
    st.subheader("🤖 AI Topics")
    topic2 = st.text_input("Enter topic", "machine learning", key="t2")

    if st.button("Generate Word Cloud", key="b2"):
        wc, msg = generate_wordcloud(topic2)
        st.info(msg)

        if wc:
            st.image(wc.to_array())


# ---------- TAB 3 ----------
with tab3:
    st.subheader("📊 Data Science")
    topic3 = st.text_input("Enter topic", "data science", key="t3")

    if st.button("Generate Word Cloud", key="b3"):
        wc, msg = generate_wordcloud(topic3)
        st.info(msg)

        if wc:
            st.image(wc.to_array())
