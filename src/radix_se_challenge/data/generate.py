"""Scripts to download the dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from radix_se_challenge.constants import DATA_F


def download(write_f: Path | None = None) -> pd.DataFrame:
    """Download the dataset."""
    write_f = write_f or DATA_F

    # Download the dataset
    download_url = requests.get(
        url="https://5wlajyc8dl.execute-api.eu-west-1.amazonaws.com/prod/train_url/2",
    ).text
    with open(write_f / "dataset.csv", "w") as f:
        f.write(requests.get(url=json.loads(download_url)).text)

    # Return the extracted dataframe
    return pd.read_csv(write_f / "dataset.csv")


def split(df: pd.DataFrame, r_test: float = 0.1, write_f: Path | None = None) -> None:
    """Split the dataset in train and test splits."""
    write_f = write_f or DATA_F
    df_train, df_test = train_test_split(df, test_size=r_test, random_state=42)
    df_train.to_csv(write_f / "train.csv")
    df_test.to_csv(write_f / "test.csv")


def main() -> None:
    """Generate the dataset splits."""
    # Download the dataset first
    my_df = download()

    # Create the splits
    split(my_df)


if __name__ == "__main__":
    main()
