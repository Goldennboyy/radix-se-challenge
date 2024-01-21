"""Model code."""

from __future__ import annotations

import pickle
from logging import warning
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from radix_se_challenge.model.utils import simple_process


class Model:
    """Custom genre predictive model."""

    def __init__(self) -> None:
        """Initialise the model."""
        self.binarizer = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer()
        self.clf = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced"),
            n_jobs=-1,
        )

    def __call__(self, df: pd.DataFrame) -> dict[int, dict[int, str]]:
        """Make predictions using the model."""
        probs = self.predict_proba(df=df)

        # Format the prediction
        submission = {}
        for i, (movie_id, pred) in enumerate(zip(df.movie_id, probs)):
            submission[movie_id] = {i: pred[i] for i in range(5)}
        return submission

    def predict_proba(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """Get the genre probabilities."""
        probs = self.clf.predict_proba(self.tfidf.transform(df.synopsis))

        # Transform probabilities to genres
        preds = []
        for args in (-probs).argsort():
            preds.append([self.binarizer.classes_[idx] for idx in args[:5]])
        return np.stack(preds)

    def train(self, df: pd.DataFrame) -> None:
        """Train the model on the provided dataframe."""
        # Collect text by genre
        all_genres = sorted({x for y in df.genres for x in y.split()})
        train_texts = {g: "" for g in all_genres}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            for g in row.genres.split():
                train_texts[g] += simple_process(row.synopsis) + " "

        # Create the TF-IDF vectorizer and fit it on the texts
        self.tfidf.fit([train_texts[g] for g in all_genres])

        # Get binary representation of targets
        y = self.binarizer.fit_transform(df["genres"].str.split())

        # Train a LinearRegression model on the given data
        self.clf.fit(self.tfidf.transform(df.synopsis), y)

    def save(self, mdl_f: Path) -> None:
        """Save the current model."""
        mdl_f.mkdir(exist_ok=True, parents=True)
        pickle.dump(self.clf, open(mdl_f / "classifier.pickle", "wb"))
        pickle.dump(self.binarizer, open(mdl_f / "binarizer.pickle", "wb"))
        pickle.dump(self.tfidf, open(mdl_f / "tfidf.pickle", "wb"))

    @classmethod
    def load(cls, mdl_f: Path) -> Model:
        """Load in a pre-trained model."""
        model = cls()
        if all(
            (mdl_f / x).is_file() for x in ("classifier.pickle", "binarizer.pickle", "tfidf.pickle")
        ):
            model.clf = pickle.load(open(mdl_f / "classifier.pickle", "rb"))
            model.binarizer = pickle.load(open(mdl_f / "binarizer.pickle", "rb"))
            model.tfidf = pickle.load(open(mdl_f / "tfidf.pickle", "rb"))
        else:
            warning("No pre-trained model found, using randomly initialed one")
        return model
