"""Test data loading and generating."""

from __future__ import annotations

from pathlib import Path

from radix_se_challenge.data.generate import download, split


class TestData:
    """Test data loading and processing."""

    def test_download_and_split(self) -> None:
        """Test the loading of the regular dataset splits."""
        my_f = Path(__file__).parent / "../data"
        my_f.mkdir(parents=True, exist_ok=True)

        # Download the dataset first
        my_df = download(write_f=my_f)

        # Create the splits
        split(my_df, write_f=my_f)
