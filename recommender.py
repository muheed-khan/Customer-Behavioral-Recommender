"""Caser sequential recommendation model, wrapped for use by the Flask app.

Trains on import so the first request after startup already has a ready model.
Training happens once per process (not per-request), which keeps the demo
responsive while still reflecting the dataset each time the server starts.
"""
import numpy as np
import tensorflow as tf
from libreco.algorithms import Caser
from libreco.data import DatasetFeat

SPARSE_COLS = [
    "brand", "label", "genre1", "genre2", "genre3", "genre4", "genre5",
    "genre6", "genre7", "genre8", "genre9", "genre10", "genre11",
    "genre12", "genre13", "genre14", "genre15", "genre16", "genre17",
    "genre18", "genre19", "genre20", "genre21", "genre22",
]
DENSE_COLS = ["rating"]
USER_COLS = []
ITEM_COLS = ["rating"] + SPARSE_COLS

MODEL_CONFIG = {
    "task": "ranking",
    "loss_type": "focal",
    "embed_size": 16,
    # 1 epoch leaves the model undertrained enough that every user/item gets
    # near-identical "most popular" recommendations (predictions dominated by
    # item popularity bias rather than the learned embeddings). 10 epochs gets
    # genuinely different recommendations per user/item at the cost of a
    # longer startup (~70-90s instead of ~8s on a typical machine).
    "n_epochs": 10,
    "lr": 3e-3,
    "lr_decay": False,
    "reg": None,
    "batch_size": 64,
    "sampler": "popular",
    "num_neg": 1,
    "use_bn": True,
    "recent_num": 10,
}


class Recommender:
    """Thin wrapper around a LibRecommender Caser model."""

    def __init__(self, data_file):
        self.data_file = data_file
        self.model = None
        self.data_info = None

    def train_model(self):
        import pandas as pd

        # Reset the TF graph in case a previous model was trained in this process.
        tf.compat.v1.reset_default_graph()

        data = pd.read_csv(self.data_file)
        train_data, self.data_info = DatasetFeat.build_trainset(
            data, USER_COLS, ITEM_COLS, SPARSE_COLS, DENSE_COLS
        )

        self.model = Caser(
            task=MODEL_CONFIG["task"],
            data_info=self.data_info,
            loss_type=MODEL_CONFIG["loss_type"],
            embed_size=MODEL_CONFIG["embed_size"],
            n_epochs=MODEL_CONFIG["n_epochs"],
            lr=MODEL_CONFIG["lr"],
            lr_decay=MODEL_CONFIG["lr_decay"],
            reg=MODEL_CONFIG["reg"],
            batch_size=MODEL_CONFIG["batch_size"],
            sampler=MODEL_CONFIG["sampler"],
            num_neg=MODEL_CONFIG["num_neg"],
            use_bn=MODEL_CONFIG["use_bn"],
            recent_num=MODEL_CONFIG["recent_num"],
            tf_sess_config={"gpu_options": {"allow_growth": True}},
        )
        self.model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True, eval_data=None)
        return self

    def recommend(self, user_id, n_rec=7, seq=None):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call `train_model()` first.")
        return self.model.recommend_user(user=user_id, n_rec=n_rec, seq=seq)

    def similar_items(self, item_id, n_rec=7):
        """"Customers who viewed this also viewed" — nearest neighbors of a single
        item in the model's learned embedding space.

        Caser's `recommend_user(seq=[item_id])` was tried for this first, but a
        sequence of just one real item (padded out to `recent_num`) washes out
        in its convolutional sequence encoder and collapses to near-identical
        "most popular" results regardless of which item is passed in. Plain
        cosine similarity between learned item embeddings does not have that
        problem and actually varies per item.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call `train_model()` first.")
        item2id = self.data_info.item2id
        if item_id not in item2id:
            return []

        item_embeds = self.model.item_embeds_np[: self.model.n_items]
        inner_id = item2id[item_id]
        target = item_embeds[inner_id]
        norms = np.linalg.norm(item_embeds, axis=1) * np.linalg.norm(target) + 1e-8
        similarities = item_embeds @ target / norms

        ranked = np.argsort(-similarities)
        top_ids = [i for i in ranked if i != inner_id][:n_rec]
        return [self.data_info.id2item[i] for i in top_ids]
