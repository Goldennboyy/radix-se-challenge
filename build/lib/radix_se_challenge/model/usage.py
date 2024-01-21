"""Usage functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radix_se_challenge.constants import MDL_F
from radix_se_challenge.model import Model
from radix_se_challenge.model.evaluation import apk


def train(df: pd.DataFrame, mdl_f: Path | None = None) -> None:
    """Train and save the model."""
    mdl = Model()
    mdl.train(df=df)
    mdl.save(mdl_f=mdl_f or MDL_F)


def predict(df: pd.DataFrame, mdl_f: Path | None = None) -> dict[int, dict[int, str]]:
    """Make predictions with the model."""
    mdl = Model.load(mdl_f=mdl_f or MDL_F)
    return mdl(df=df)


def evaluate(df: pd.DataFrame, mdl_f: Path | None = None) -> dict[str, float]:
    """Evaluate the model."""
    # Make the prediction
    results = predict(df, mdl_f=mdl_f or MDL_F)
    assert set(df.movie_id) == set(results.keys())

    # Flatten the targets and results
    keys, targets, preds = [], [], []
    for _, row in df.iterrows():
        keys.append(row.movie_id)
        targets.append(row.genres.split(" "))
        preds.append([results[row.movie_id][i] for i in range(5)])

    # Calculate the score
    scores = sorted([apk(actual=t, predicted=p, k=5) for t, p in zip(targets, preds)])
    result = {
        "mean_ap5": sum(scores) / len(scores),
        "median_ap5": scores[len(scores) // 2],
    }
    print(f"Mean Average Precision @5: {100*result['mean_ap5']:.2f}%")
    print(f"Median Average Precision @5: {100*result['median_ap5']:.2f}%")
    return result


if __name__ == "__main__":
    from radix_se_challenge.data import load_train_test

    # Load in the datasets
    df_train, df_test = load_train_test()

    # Train the model
    train(df_train)

    # Evaluate the model
    _ = evaluate(df_test)

