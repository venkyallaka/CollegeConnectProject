"""
common.py
=========
Shared preprocessing + train/save/load helpers so every model folder
(placement, cgpa, scholarship, mentoring, internship) follows the exact
same pattern with minimal duplicated code.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ft_transformer import FTTransformer


class TaskBundle:
    """Holds everything needed to train, save, load and run one FT-Transformer task."""

    def __init__(self, name: str, continuous_cols: list[str], categorical_cols: list[str],
                 target_col: str, task_type: str):
        self.name = name
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.task_type = task_type  # "classification" or "regression"

        self.scaler = StandardScaler()
        self.cat_encoders = {c: LabelEncoder() for c in categorical_cols}
        self.target_encoder = LabelEncoder() if task_type == "classification" else None
        self.model: FTTransformer | None = None

    # ---------- preprocessing ----------
    def fit_transform(self, df: pd.DataFrame):
        x_cont = self.scaler.fit_transform(df[self.continuous_cols].astype(float)) if self.continuous_cols else np.zeros((len(df), 0))
        x_cat = np.column_stack([
            self.cat_encoders[c].fit_transform(df[c].astype(str)) for c in self.categorical_cols
        ]) if self.categorical_cols else np.zeros((len(df), 0), dtype=int)

        if self.task_type == "classification":
            y = self.target_encoder.fit_transform(df[self.target_col].astype(str)).astype(np.float32)
        else:
            y = df[self.target_col].astype(float).to_numpy(dtype=np.float32)

        return x_cont.astype(np.float32), x_cat.astype(np.int64), y

    def transform(self, df: pd.DataFrame):
        x_cont = self.scaler.transform(df[self.continuous_cols].astype(float)) if self.continuous_cols else np.zeros((len(df), 0))
        x_cat = np.column_stack([
            self.cat_encoders[c].transform(df[c].astype(str)) for c in self.categorical_cols
        ]) if self.categorical_cols else np.zeros((len(df), 0), dtype=int)
        return x_cont.astype(np.float32), x_cat.astype(np.int64)

    def cardinalities(self):
        return [len(self.cat_encoders[c].classes_) for c in self.categorical_cols]

    # ---------- train ----------
    def train(self, df: pd.DataFrame, epochs: int = 8, batch_size: int = 512, lr: float = 1e-3,
              d_token: int = 24, n_heads: int = 4, n_blocks: int = 2, verbose: bool = True):
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

        x_cont, x_cat, y = self.fit_transform(train_df)
        xv_cont, xv_cat = self.transform(val_df)
        if self.task_type == "classification":
            yv = self.target_encoder.transform(val_df[self.target_col].astype(str)).astype(np.float32)
        else:
            yv = val_df[self.target_col].astype(float).to_numpy(dtype=np.float32)

        n_outputs = 1 if self.task_type == "regression" else (
            1 if len(self.target_encoder.classes_) == 2 else len(self.target_encoder.classes_)
        )

        self.model = FTTransformer(
            n_continuous=len(self.continuous_cols),
            categorical_cardinalities=self.cardinalities(),
            d_token=d_token, n_heads=n_heads, n_blocks=n_blocks,
            n_outputs=n_outputs,
        )

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = (
            torch.nn.BCEWithLogitsLoss() if (self.task_type == "classification" and n_outputs == 1)
            else (torch.nn.CrossEntropyLoss() if self.task_type == "classification" else torch.nn.MSELoss())
        )

        x_cont_t, x_cat_t, y_t = torch.tensor(x_cont), torch.tensor(x_cat), torch.tensor(y)
        xv_cont_t, xv_cat_t, yv_t = torch.tensor(xv_cont), torch.tensor(xv_cat), torch.tensor(yv)

        n = x_cont_t.shape[0]
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                bc, bcat, by = x_cont_t[idx], x_cat_t[idx], y_t[idx]
                opt.zero_grad()
                out = self.model(bc, bcat)
                if self.task_type == "classification" and n_outputs > 1:
                    loss = loss_fn(out, by.long())
                else:
                    loss = loss_fn(out, by)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            avg_loss = total_loss / n

            self.model.eval()
            with torch.no_grad():
                val_out = self.model(xv_cont_t, xv_cat_t)
                if self.task_type == "classification" and n_outputs > 1:
                    val_loss = loss_fn(val_out, yv_t.long()).item()
                    val_metric = (val_out.argmax(dim=-1) == yv_t.long()).float().mean().item()
                    metric_name = "val_acc"
                elif self.task_type == "classification":
                    val_loss = loss_fn(val_out, yv_t).item()
                    val_metric = ((torch.sigmoid(val_out) > 0.5).float() == yv_t).float().mean().item()
                    metric_name = "val_acc"
                else:
                    val_loss = loss_fn(val_out, yv_t).item()
                    val_metric = val_loss ** 0.5
                    metric_name = "val_rmse"

            if verbose:
                print(f"[{self.name}] epoch {epoch + 1}/{epochs}  train_loss={avg_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  {metric_name}={val_metric:.4f}")

        return self

    # ---------- predict ----------
    def predict(self, df: pd.DataFrame):
        x_cont, x_cat = self.transform(df)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(x_cont), torch.tensor(x_cat))
        if self.task_type == "regression":
            return out.numpy()
        if out.ndim == 1:  # binary, single logit
            probs = torch.sigmoid(out).numpy()
            labels = self.target_encoder.inverse_transform((probs > 0.5).astype(int))
            return labels, probs
        else:
            probs = torch.softmax(out, dim=-1).numpy()
            labels = self.target_encoder.inverse_transform(probs.argmax(axis=-1))
            return labels, probs

    # ---------- save / load ----------
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(folder, "model.pt"))

        meta = {
            "name": self.name,
            "continuous_cols": self.continuous_cols,
            "categorical_cols": self.categorical_cols,
            "target_col": self.target_col,
            "task_type": self.task_type,
            "cardinalities": self.cardinalities(),
            "d_token": self.model.tokenizer.d_token,
            "scaler_mean": self.scaler.mean_.tolist() if self.continuous_cols else [],
            "scaler_scale": self.scaler.scale_.tolist() if self.continuous_cols else [],
            "cat_classes": {c: self.cat_encoders[c].classes_.tolist() for c in self.categorical_cols},
            "target_classes": self.target_encoder.classes_.tolist() if self.target_encoder else None,
            "n_outputs": self.model.head[-1].out_features,
            "n_heads": self.model.encoder.layers[0].self_attn.num_heads,
            "n_blocks": len(self.model.encoder.layers),
        }
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved '{self.name}' model -> {folder}/ (model.pt + meta.json)")

    @classmethod
    def load(cls, folder: str) -> "TaskBundle":
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)

        bundle = cls(meta["name"], meta["continuous_cols"], meta["categorical_cols"],
                     meta["target_col"], meta["task_type"])

        if bundle.continuous_cols:
            bundle.scaler.mean_ = np.array(meta["scaler_mean"])
            bundle.scaler.scale_ = np.array(meta["scaler_scale"])
            bundle.scaler.var_ = bundle.scaler.scale_ ** 2
            bundle.scaler.n_features_in_ = len(bundle.continuous_cols)

        for c in bundle.categorical_cols:
            bundle.cat_encoders[c].classes_ = np.array(meta["cat_classes"][c])

        if meta["target_classes"] is not None:
            bundle.target_encoder.classes_ = np.array(meta["target_classes"])

        bundle.model = FTTransformer(
            n_continuous=len(meta["continuous_cols"]),
            categorical_cardinalities=meta["cardinalities"],
            d_token=meta["d_token"], n_heads=meta["n_heads"], n_blocks=meta["n_blocks"],
            n_outputs=meta["n_outputs"],
        )
        bundle.model.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location="cpu"))
        bundle.model.eval()
        return bundle

