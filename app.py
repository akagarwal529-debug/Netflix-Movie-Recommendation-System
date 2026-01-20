import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #f9fafb;
}
.stButton>button {
    background-color: #e50914;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}
.recommend-box {
    background-color: #020617;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

with open("netflix_recommendation_model.pkl", "rb") as file:
    model = pickle.load(file)

df = model["dataframe"]
tfidf = model["tfidf_vectorizer"]

st.markdown("<h1 style='text-align:center;'>🎬 Netflix Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Content-Based Filtering</p>", unsafe_allow_html=True)

movie_name = st.selectbox("Select a Movie", sorted(df['title'].unique()))

if st.button("Recommend Movies"):
    tfidf_matrix = tfidf.transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    idx = df[df['title'] == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("🎯 Recommended Movies")
    for i in sim_scores:
        st.markdown(
            f"<div class='recommend-box'>🍿 {df['title'].iloc[i[0]]}</div>",
            unsafe_allow_html=True
        )
