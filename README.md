# 🥗 Smart Eats — AI-Powered Restaurant Recommender

Smart Eats is a lightweight AI engineering demo that simulates an Uber Eats-style food-recommendation system.

Save your time in finding target restaurant in your city!

In just a few clicks, users can enter their city, food craving, budget, and optional dietary restriction to get AI-generated Top-5 restaurant picks with explanations.

![NY_salad_vegan](https://github.com/user-attachments/assets/3ca2004d-7c48-414b-a997-0a5cf8197307)

# 🚀 Quick Start
```bash
pip install -r requirements.txt
python -m src.data_prep
python -m src.embed_index
streamlit run app/app_streamlit.py
```

Place the restaurant datasets (or demo Kaggle CSVs) inside the data/ folder.

The app runs fully locally; with an OpenAI API key, GPT will re-rank and explain recommendations.

# 💡 Key Features

**End-to-end AI engineering pipeline:** data prep → embedding → model re-ranking → web UI

**GPT-based reasoning:** natural-language explanations for each recommendation

**Fallback mode:** works offline using heuristic scoring when API quota = 0

**Streamlit UI:** clean interface for quick testing
