import random
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from recommender import Recommender

BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_FILE = BASE_DIR / "data" / "products.json"
RATINGS_FILE = BASE_DIR / "data" / "final.csv"

CATALOG_SIZE = 30   # products shown in the "click to test" grid
NUM_SAMPLE_SHOPPERS = 20   # entries in the "simulate as shopper" dropdown
N_RECOMMENDATIONS = 8

app = Flask(__name__)

df = None
catalog = None
demo_shoppers = None
recommender = None


def _first_image(images):
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, list) and first:
            return first[0]
        if isinstance(first, str):
            return first
    return None


def load_data():
    """Load the product catalog and prepare display fields.

    Product photos are served straight from their original Amazon CDN URL
    (`imageURLHighRes`) instead of shipping the ~800MB local image dump that
    was used while building the dataset — keeps the repo/container small.
    """
    try:
        data = pd.read_json(PRODUCTS_FILE, lines=True)

        def resolve_image(row):
            local = _first_image(row.get("image_filename"))
            remote = _first_image(row.get("imageURLHighRes"))
            if remote:
                return remote
            if local:
                return "/static/" + local
            return "/static/img/default.jpg"

        data["image_filename"] = data.apply(resolve_image, axis=1)
        data["price"] = data["price"].apply(lambda x: x if pd.notnull(x) else random.randint(100, 500))
        data["asin"] = data["asin"].astype(str)
        data["product_id"] = data.index
        data["name"] = data["predicted"].fillna("product").str.title()
        return data
    except Exception as e:
        print(f"Error loading product data: {e}")
        return pd.DataFrame()


def load_demo_shoppers(n=NUM_SAMPLE_SHOPPERS):
    """Pick a small, fixed sample of real users for the 'simulate as shopper' demo.

    The real dataset has ~25k user ids, which is unusable in a dropdown — this
    gives a stable, friendly-labeled subset instead.
    """
    try:
        data = pd.read_csv(RATINGS_FILE)
        all_users = sorted(data["user"].dropna().unique().tolist())
        sample = random.Random(42).sample(all_users, min(n, len(all_users)))
        return [{"id": uid, "label": f"Sample Shopper {i + 1}"} for i, uid in enumerate(sample)]
    except Exception as e:
        print(f"Error loading users: {e}")
        return []


def init_app():
    global df, catalog, demo_shoppers, recommender

    recommender = Recommender(str(RATINGS_FILE))
    recommender.train_model()

    df = load_data()
    demo_shoppers = load_demo_shoppers()
    catalog = df.sample(n=min(CATALOG_SIZE, len(df)), random_state=42) if len(df) else df


init_app()


def recommend_from_product(product):
    """'Because you viewed this' — nearest neighbors of the clicked item in the
    model's learned embedding space (see Recommender.similar_items)."""
    try:
        asins = recommender.similar_items(product["asin"], n_rec=N_RECOMMENDATIONS)
        return df[df["asin"].isin(asins)]
    except Exception as e:
        print(f"Error generating item-based recommendations: {e}")
        return df.sample(min(N_RECOMMENDATIONS, len(df)))


def recommend_for_user(user_id):
    """Personalized picks for a real user, based on their purchase history."""
    try:
        asins = recommender.recommend(user_id, n_rec=N_RECOMMENDATIONS)[user_id]
        return df[df["asin"].isin(asins)]
    except Exception as e:
        print(f"Error generating user-based recommendations: {e}")
        return df.sample(min(N_RECOMMENDATIONS, len(df)))


@app.route("/")
def index():
    selected_product = None
    selected_shopper = None
    recommendations = None

    user_id = request.args.get("user") or None
    product_id = request.args.get("product", type=int)

    if product_id is not None and (df["product_id"] == product_id).any():
        selected_product = df[df["product_id"] == product_id].iloc[0].to_dict()
        recommendations = recommend_from_product(selected_product).to_dict(orient="records")
    elif user_id:
        selected_shopper = next((s for s in demo_shoppers if s["id"] == user_id), None)
        if selected_shopper:
            recommendations = recommend_for_user(user_id).to_dict(orient="records")

    return render_template(
        "index.html",
        catalog=catalog.to_dict(orient="records"),
        shoppers=demo_shoppers,
        selected_product=selected_product,
        selected_shopper=selected_shopper,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True)
