# embed_index placeholder
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
META = DATA / "meta.parquet"
EMB = DATA / "embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"  # fast + decent


def build_index():
    print("[embed_index] Loading meta…")
    meta = pd.read_parquet(META)
    model = SentenceTransformer(MODEL_NAME)
    print("[embed_index] Encoding descriptions…")
    X = model.encode(meta["desc"].fillna("").tolist(), normalize_embeddings=True)
    np.save(EMB, X)
    print(f"[embed_index] Saved {EMB} with shape {X.shape}")


def _hard_filter(df, city=None, price_max=None, dietary=None):
    sub = df.copy()
    if city:
        sub = sub[sub["city"].str.contains(str(city), case=False, na=False)]
    if price_max is not None and "mid_price" in sub.columns:
        sub = sub[pd.to_numeric(sub["mid_price"], errors="coerce") <= float(price_max)]
    if dietary:
        sub = sub[sub.get("dietary_tags", "").astype(str).str.contains(dietary, case=False, na=False)]
    return sub


def search(query: str, topk: int = 30, city: str = None, price_max: float = None, dietary: str = None):
    """Return topk candidates after hard filtering by cosine similarity."""
    meta = pd.read_parquet(META)
    X = np.load(EMB)
    model = SentenceTransformer(MODEL_NAME)
    qv = model.encode([query], normalize_embeddings=True)

    sub = _hard_filter(meta, city, price_max, dietary)
    if len(sub) == 0:
        return sub

    idx = sub.index.to_numpy()
    sims = cosine_similarity(qv, X[idx])[0]
    order = np.argsort(sims)[-topk:][::-1]
    return sub.iloc[order].assign(score=sims[order])


if __name__ == "__main__":
    build_index()