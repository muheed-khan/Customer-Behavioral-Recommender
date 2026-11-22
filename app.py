import random
from pathlib import Path

import pandas as pd
from flask import Flask, render_template

from recommender import Recommender

BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_FILE = BASE_DIR / "data" / "products.json"
RATINGS_FILE = BASE_DIR / "data" / "final.csv"

app = Flask(__name__)

df = None
featured_products = None
user_list = None
selected_user = None
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
    was used while building the dataset — this keeps the repo/container small
    and deployable on free hosting tiers.
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
        return data
    except Exception as e:
        print(f"Error loading product data: {e}")
        return pd.DataFrame()


def load_users():
    try:
        data = pd.read_csv(RATINGS_FILE)
        return sorted(data["user"].dropna().unique().tolist())
    except Exception as e:
        print(f"Error loading users: {e}")
        return []


def init_app():
    global df, featured_products, user_list, recommender

    recommender = Recommender(str(RATINGS_FILE))
    recommender.train_model()

    df = load_data()
    user_list = load_users()
    featured_products = df.sample(n=min(12, len(df))) if len(df) else df


init_app()


@app.route("/")
def home():
    return render_template(
        "index.html",
        products=featured_products.to_dict(orient="records"),
        users=user_list,
        selected_user=selected_user,
    )


@app.route("/shop/")
@app.route("/shop/<int:page>")
def shop(page=1):
    products_per_page = 18
    start_idx = (page - 1) * products_per_page
    paginated_products = df.iloc[start_idx:start_idx + products_per_page]

    total_pages = (len(df) + products_per_page - 1) // products_per_page
    pagination = list(range(1, total_pages + 1)) if total_pages <= 7 else (
        [1, "...", page - 1, page, page + 1, "...", total_pages] if 4 <= page <= total_pages - 3 else
        ([1, 2, 3, 4, 5, "...", total_pages] if page < 4 else
         [1, "...", total_pages - 4, total_pages - 3, total_pages - 2, total_pages - 1, total_pages])
    )

    return render_template(
        "shop.html",
        products=paginated_products.to_dict(orient="records"),
        pagination=pagination,
        current_page=page,
        selected_user=selected_user,
        users=user_list,
    )


@app.route("/recommendations/<string:user_id>")
def recommendations(user_id):
    global selected_user
    selected_user = user_id

    recommended_asins = recommender.recommend(selected_user)
    recommended_products = df[df["asin"].isin(recommended_asins)].copy()

    return render_template(
        "recommendations.html",
        recommended_products=recommended_products.to_dict(orient="records"),
        users=user_list,
        selected_user=selected_user,
    )


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = df[df["product_id"] == product_id].iloc[0].to_dict()
    recommended_products = df.sample(min(7, len(df))).to_dict(orient="records")

    return render_template(
        "details.html",
        product=product,
        recommendations=recommended_products,
        selected_user=selected_user,
        users=user_list,
    )


@app.route("/wishlist")
def wishlist():
    wishlist_items = df.sample(min(5, len(df)))
    return render_template(
        "wishlist.html",
        wishlist_items=wishlist_items.to_dict(orient="records"),
        selected_user=selected_user,
        users=user_list,
    )


if __name__ == "__main__":
    app.run(debug=True)
