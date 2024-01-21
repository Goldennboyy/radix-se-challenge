"""Test complete pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from radix_se_challenge.data import load_train_test
from radix_se_challenge.model import evaluate, train


class TestPipeline:
    """Test data loading and processing."""

    @pytest.fixture(scope="class")
    def datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load in the datasets used for training."""
        df_train, df_test = load_train_test()
        return df_train[:1_000], df_test[:100]  # Use only a sample

    def test_pipeline(self, datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test the loading of the regular dataset splits."""
        df_train, df_test = datasets
        my_f = Path(__file__).parent / "../data"

        # Train the model
        train(df_train, mdl_f=my_f)

        # Evaluate the model
        evaluate(df_test, mdl_f=my_f)
