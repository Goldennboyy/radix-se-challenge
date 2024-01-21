"""Dataset loading."""

from __future__ import annotations

import pandas as pd

from radix_se_challenge.constants import DATA_F
from radix_se_challenge.data.generate import main as generate_datasets


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load in the train and test datasets."""
    # Assure datasets exist
    if not ((DATA_F / "train.csv").is_file() and (DATA_F / "test.csv").is_file()):
        generate_datasets()

    # Load in the datasets
    df_train = pd.read_csv(DATA_F / "train.csv", index_col=0)
    df_test = pd.read_csv(DATA_F / "test.csv", index_col=0)
    return df_train, df_test
