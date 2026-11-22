# Recommendation Engine Tester

A single-page Flask demo built to let you **try and judge** a deep-learning
sequential recommender (the **Caser** model, via
[LibRecommender](https://github.com/massquantity/LibRecommender)) trained on a
real Amazon product-interactions dataset. No storefront, no login, no
clutter — just two ways to test it:

1. **Click any product** in the grid — the model recommends similar items
   based on that single click (shown directly above the grid, with the
   clicked product highlighted). Click a recommended product to chain into
   its recommendations, and so on.
2. **Pick a sample shopper** from the dropdown — see what the model
   recommends for someone with real purchase history.

> This project was originally developed and trained inside the **Spyder IDE**
> (hence folders like `.spyder-py3` in earlier revisions). This repo is a
> cleaned-up, standalone version of just the parts needed to run the web app
> — it has no dependency on Spyder and runs from any terminal, Docker, or
> hosting platform. See [Running it from Spyder](#running-it-from-spyder-optional)
> if you still want to open it there.

## How it works

- `recommender.py` trains a **Caser** model (a CNN-based sequential
  recommender) on `data/final.csv` (user/item interactions + item metadata)
  using LibRecommender, with TensorFlow as the backend.
- `app.py` is a single-route Flask app (`/`) that trains the model once at
  startup, loads the product catalog from `data/products.json`, and handles
  two optional query params:
  - `?product=<id>` — "because you viewed this" recommendations for one item
    (works for any item, no known user required — a cold-start session).
  - `?user=<id>` — personalized recommendations for one of a small, fixed
    sample of real users from the dataset (shown with friendly labels like
    "Sample Shopper 1" instead of raw Amazon reviewer ids).
- Product photos are served directly from their original Amazon CDN URLs
  (already present in the dataset) instead of bundling a ~800MB local image
  cache, which keeps the repo small and deployable on free hosting tiers.

## Project structure

```
product-recommender/
├── app.py                 # Flask app (single route, see "How it works")
├── recommender.py          # Caser model training/inference wrapper
├── requirements.txt         # Pinned Python dependencies
├── Dockerfile               # Container build (used for most hosts)
├── docker-compose.yml       # One-command local Docker run
├── Procfile                  # For Render/Railway/Heroku-style platforms
├── data/
│   ├── final.csv             # User-item interactions + item features (training data)
│   └── products.json         # Product catalog
├── static/img/default.jpg    # Fallback image for products with no photo
└── templates/index.html      # The entire UI (self-contained, no build step)
```

## Quickstart

### Option A — Docker (recommended, fully reproducible)

Requires only [Docker](https://docs.docker.com/get-docker/) installed — no
Python setup needed.

```bash
docker compose up --build
```

Then open **http://localhost:8000**. First boot trains the model, which
takes roughly 10–30 seconds depending on your machine.

### Option B — Local Python environment

Requires **Python 3.11** (the version this was tested against — TensorFlow
and LibRecommender are picky about Python/TF version combinations, see
[Why these exact versions?](#why-these-exact-versions)).

```bash
python3.11 -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**.

## Running it from Spyder (optional)

If you prefer the original Spyder workflow:

1. Open this folder as your Spyder project / working directory.
2. Create/select an interpreter with the packages from `requirements.txt`
   installed (Spyder → Preferences → Python interpreter, or install into
   Spyder's own environment with `conda install`/`pip install` from the
   Spyder console).
3. Open `app.py` and press **Run** (F5). The console will show training
   progress, then `Running on http://127.0.0.1:5000`.

All file paths in `app.py` are resolved relative to the script's own
location (`Path(__file__).resolve().parent`), so it runs correctly no
matter what Spyder's current working directory is set to.

## Deploying it for free

**Netlify will not work for this project** — it only hosts static sites and
small serverless functions, not a long-lived Python/Flask process with
TensorFlow/PyTorch loaded in memory.

Measured on this project: dependencies take ~2.8GB on disk, and the app uses
~650–700MB RAM once the model is trained and serving. **Use
[Hugging Face Spaces](https://huggingface.co/spaces)** (free CPU tier, 16GB
RAM — plenty of headroom, and it's built for ML demos like this one):

1. Push this folder to a GitHub repo (see below).
2. Create a new Space at huggingface.co/spaces → SDK: **Docker** → link it to
   your GitHub repo (or push directly to the Space's own git remote).
3. It builds from the `Dockerfile` as-is — no extra config needed.
4. First build installs TensorFlow/PyTorch (~5–10 min); the model trains
   again on every cold start/restart since nothing is cached between boots.

## Pushing to GitHub

```bash
cd product-recommender
git init
git add .
git commit -m "Initial commit: Caser-based product recommender demo"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Why these exact versions?

LibRecommender's Caser implementation needs `tensorflow<2.16` (newer
TensorFlow removed APIs it depends on) and also imports PyTorch-based
algorithms at package-import time, so `torch` must be installed even though
this project only uses the TensorFlow-based Caser model. `requirements.txt`
pins a tested, compatible combination:

- Python 3.11
- TensorFlow 2.15.1
- PyTorch 2.1.2 (CPU-only build, via the `--extra-index-url` in
  `requirements.txt` — keeps the image a few hundred MB instead of several GB)
- LibRecommender 1.5.2

Using different versions may work, but this is the combination that was
actually installed and tested end-to-end for this repo.

## Troubleshooting

- **Port already in use**: change the port in `docker-compose.yml`
  (`"8000:8000"` → `"8080:8000"`) or stop whatever else is using it.
- **Training is slow / times out on a free host**: the model trains once per
  process start (not per request). On very low-memory free tiers, reduce
  `embed_size`/`batch_size` in `recommender.py`'s `MODEL_CONFIG` if needed.
- **`pip install` fails on an unusual Python version**: this project is
  tested on Python 3.11 specifically — LibRecommender only ships prebuilt
  wheels for Python 3.6–3.11.

## Dataset & credits

`data/final.csv` and `data/products.json` are derived from a public Amazon
product reviews dataset, enriched with category/genre labels used to train
the Caser model.
