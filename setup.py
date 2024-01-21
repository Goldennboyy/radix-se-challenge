"""Setup module for this Python package."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).parent / "requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and line[0] != "#"]

setup(
    name="radix_se_challenge",
    version="0.0.0",
    description="Hiring challenge for the Software Engineer applicants at Radix.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    include_package_data=True,
)
