# ---- add project root to sys.path ----
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------


# streamlit app placeholder
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.embed_index import search
from src.recommend import rerank_with_gpt, rerank_fallback, _gpt_available

st.set_page_config(page_title="Smart Eats", page_icon="🍔", layout="wide")
st.title("🍔 Smart Eats — Top‑5 Recommender")

DATA = Path("data")
META = DATA / "meta.parquet"
EMB = DATA / "embeddings.npy"

with st.sidebar:
    st.header("Query")
    city  = st.text_input("City", "Chicago")
    query = st.text_input("What do you feel like eating?", "spicy noodles")
    price = st.number_input("Max budget (USD per person)", min_value=5, max_value=200, value=20, step=1)
    diet  = st.selectbox("Dietary restriction (optional)", ["", "vegan","vegetarian","gluten-free","halal","kosher","nut-free"]) or None
    k     = st.slider("Candidates to consider (Top‑K)", 10, 50, 30)
    run   = st.button("Recommend")

if run:
    if not META.exists() or not EMB.exists():
        st.error("Index not built. Run: `python -m src.data_prep` then `python -m src.embed_index`. ")
        st.stop()

    # Retrieve Top‑K by cosine similarity with hard filters
    cands = search(query, topk=k, city=city, price_max=price, dietary=diet)
    if len(cands) == 0:
        st.warning("No matches after hard filtering. Try relaxing budget/dietary or a broader query.")
        st.stop()

    st.caption(f"Candidates after retrieval: {len(cands)}")

    # Re‑rank + explain
    profile = {"city": city, "query": query, "price_max": price, "dietary": diet}
    if _gpt_available():
        picks = rerank_with_gpt(profile, cands.to_dict(orient="records"))
    else:
        st.info("OPENAI_API_KEY not set → using local fallback explanations.")
        picks = rerank_fallback(profile, cands.to_dict(orient="records"))

    # Render Top‑5 cards
    st.subheader("Top‑5 Picks")
    for i, p in enumerate(picks, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {p['restaurant_name']}**  ")
            st.write(p.get("justification", ""))
            col1, col2, col3 = st.columns(3)
            col1.metric("Est. cost", f"${int(p.get('estimated_cost_per_person',0))}")
            reasons = p.get("match_reasons", [])
            col2.write("**Reasons**: " + ", ".join(reasons))
            col3.write(f"**Confidence**: {p.get('confidence',0):.2f}")

    # Simple scatter of rating vs price for the candidate set
    if {"mid_price", "avg_rating"}.issubset(cands.columns):
        st.subheader("Candidate Landscape (Rating vs Price)")
        plot_df = cands[["restaurant_name","mid_price","avg_rating","score"]].dropna()
        st.scatter_chart(plot_df, x="mid_price", y="avg_rating")