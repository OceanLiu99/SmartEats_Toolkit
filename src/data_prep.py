# data_prep placeholder
import pandas as pd
import re
from pathlib import Path

DATA = Path("data")
RESTAURANTS = DATA / "restaurants.csv"
MENUS = DATA / "restaurant-menus.csv"
TRIP = DATA / "TripAdvisor_RestauarantRecommendation.csv"

OUT_META = DATA / "meta.parquet"

CITY_COL_CANDIDATES = ["city", "Location", "full_address"]

PRICE_MAP = {
    "$": 12,
    "$$": 22,
    "$$$": 40,
    "$$$$": 70,
}

DIETARY_KEYWORDS = {
    "vegan": ["vegan", "plant-based"],
    "vegetarian": ["vegetarian", "ovo", "lacto"],
    "gluten-free": ["gluten free", "gf"],
    "halal": ["halal"],
    "kosher": ["kosher"],
    "nut-free": ["nut free", "no nuts"],
}


def _extract_city_from_address(addr: str) -> str:
    if not isinstance(addr, str):
        return None
    # Try TripAdvisor style: "City, ST 12345" or "City, State"
    parts = [p.strip() for p in addr.split(',') if p.strip()]
    if len(parts) >= 1:
        # Heuristic: first part is city for TripAdvisor; for restaurants.csv, city often appears before state
        return parts[0]
    return None


def _mid_price(val) -> float:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in PRICE_MAP:
        return PRICE_MAP[s]
    # if like "$15-$25" or "$10"
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def build_meta():
    print("[data_prep] Loading CSVs…")
    rest = pd.read_csv(RESTAURANTS)
    menus = pd.read_csv(MENUS)
    trip = pd.read_csv(TRIP)

    # Basic normalize names/ids
    rest.rename(columns={"name": "restaurant_name"}, inplace=True)

    # City
    rest["city"] = rest.get("full_address", "").apply(_extract_city_from_address)
    if "Location" in trip.columns:
        trip["city"] = trip["Location"].apply(_extract_city_from_address)

    # Rating from TripAdvisor (avg)
    rating_cols = [c for c in trip.columns if c.lower().startswith("rating") or c.lower()=="rating"]
    if rating_cols:
        trip["avg_rating"] = pd.to_numeric(trip[rating_cols[0]], errors="coerce")
    else:
        trip["avg_rating"] = None

    # Reduce TripAdvisor to (name, city, avg_rating)
    trip_norm = trip.rename(columns={"Name": "restaurant_name"})
    trip_norm = trip_norm[[c for c in ["restaurant_name", "city", "avg_rating"] if c in trip_norm.columns]]

    # Price bucket
    rest["mid_price"] = rest.get("price_range", "").apply(_mid_price)

    # Cuisine/category
    rest["cuisines"] = rest.get("category")

    # Menu aggregation per restaurant (top items)
    menu_top = None
    if {"restaurant_name", "item_name"}.issubset(set(menus.columns)):
        menu_top = (
            menus.groupby("restaurant_name")["item_name"]
            .apply(lambda s: ", ".join(s.dropna().astype(str).head(5)))
            .reset_index(name="top_items")
        )

    # Merge all
    meta = rest.copy()
    if menu_top is not None:
        meta = meta.merge(menu_top, on="restaurant_name", how="left")
    if set(["restaurant_name", "city"]).issubset(trip_norm.columns):
        meta = meta.merge(trip_norm, on=["restaurant_name", "city"], how="left")

    # Dietary tags from menu text
    def extract_tags(row):
        text = " ".join([str(row.get("top_items", "")), str(row.get("category", ""))]).lower()
        tags = []
        for tag, kws in DIETARY_KEYWORDS.items():
            if any(kw in text for kw in kws):
                tags.append(tag)
        return ",".join(sorted(set(tags))) if tags else None

    meta["dietary_tags"] = meta.apply(extract_tags, axis=1)

    # Build short description
    def make_desc(r):
        bits = [
            str(r.get("restaurant_name", "")),
            f"Cuisines: {r.get('cuisines','')}",
            f"Price: ${r.get('mid_price','?')}",
            f"Top: {r.get('top_items','')}",
            f"Tags: {r.get('dietary_tags','')}",
            f"Rating: {r.get('avg_rating','?')}"
        ]
        return " | ".join([b for b in bits if b and b != ' | '])

    meta["desc"] = meta.apply(make_desc, axis=1)

    # Select minimal columns for runtime
    keep = [
      "id","restaurant_name","city","mid_price","avg_rating","cuisines","top_items",
      "dietary_tags","desc","lat","lng"
    ]
    keep = [c for c in keep if c in meta.columns]
    meta = meta[keep].drop_duplicates()

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT_META, index=False)
    print(f"[data_prep] Saved {OUT_META} with {len(meta)} rows.")


if __name__ == "__main__":
    build_meta()