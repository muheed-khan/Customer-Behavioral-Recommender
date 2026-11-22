"""Caser sequential recommendation model, wrapped for use by the Flask app.

Trains on import so the first request after startup already has a ready model.
Training happens once per process (not per-request), which keeps the demo
responsive while still reflecting the dataset each time the server starts.
"""
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
    "n_epochs": 1,
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
