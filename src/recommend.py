# recommend placeholder
import json
import os
from typing import List, Dict

import pandas as pd
from openai import OpenAI

from src.schema import ReRankSchema


def _gpt_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


# src/recommend.py
import json
from typing import List, Dict
from openai import OpenAI

def rerank_with_gpt(user_profile: Dict, candidates: List[Dict]):
    """
    Export JSON，
    """
    client = OpenAI()

    system = (
        "You are a helpful restaurant recommender. "
        "Honor hard constraints first (dietary, price, time). "
        "Return exactly 5 diverse options with brief justifications. "
        "Output STRICT JSON ONLY with key 'choices' (no extra text)."
    )

    # whitelist
    whitelist = ["id","restaurant_name","desc","mid_price","avg_rating","city","lat","lng","score"]
    small = [{k: c.get(k) for k in whitelist if k in c} for c in candidates]
    payload = {"user_profile": user_profile, "candidates": small}

    # JSON structure
    shape_hint = {
        "choices": [{
            "restaurant_id": "string",
            "restaurant_name": "string",
            "estimated_cost_per_person": "number",
            "justification": "string",
            "match_reasons": ["string"],
            "confidence": "number"
        }]*5
    }

    # using Responses API
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Follow this JSON shape strictly (no extra commentary):\n"
                    + json.dumps(shape_hint, ensure_ascii=False)
                    + "\n\nUser+Candidates:\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
            }
        ],
        temperature=0.2,
    )

    #output text
    try:
        text = resp.output_text
    except Exception:
        text = resp.output[0].content[0].text

    # clean ```json code fences
    cleaned = (
        text.strip()
        .removeprefix("```json").removeprefix("```")
        .removesuffix("```").strip()
    )

    data = json.loads(cleaned)
    return data["choices"]


def rerank_fallback(user_profile: Dict, candidates: List[Dict]):
    """No API key? Do a deterministic sort and synthesize short reasons."""
    df = pd.DataFrame(candidates)
    # Prefer higher score, higher rating, lower price
    for c in ["score", "avg_rating"]:
        if c not in df.columns:
            df[c] = 0
    if "mid_price" not in df.columns:
        df["mid_price"] = 999
    df = df.sort_values(["score", "avg_rating", "mid_price"], ascending=[False, False, True]).head(5)

    picks = []
    for _, r in df.iterrows():
        reasons = []
        if "avg_rating" in r and pd.notna(r["avg_rating"]):
            reasons.append(f"good rating {r['avg_rating']}")
        if "mid_price" in r and pd.notna(r["mid_price"]):
            reasons.append(f"~${int(r['mid_price'])} per person")
        if user_profile.get("dietary"):
            reasons.append(f"may support {user_profile['dietary']}")
        picks.append({
            "restaurant_id": str(r.get("id", "")),
            "restaurant_name": r.get("restaurant_name", "Unknown"),
            "estimated_cost_per_person": float(r.get("mid_price", 0) or 0),
            "justification": f"Match for '{user_profile.get('query','')}'. "
                              f"Reasons: {', '.join(reasons)}.",
            "match_reasons": reasons,
            "confidence": 0.55
        })
    return picks


def recommend_top5(query: str, city: str, price_max: float = None, dietary: str = None):
    """Orchestrate: retrieval (done in app) → re-rank + explain → return list of dicts."""
    raise NotImplementedError("This function is orchestrated from the Streamlit app after retrieval.")
