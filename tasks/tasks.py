"""Main tasks."""

from __future__ import annotations

import logging
import os

from invoke import task

logger = logging.getLogger(__name__)
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@task
def lint(c):
    """Lint this package."""
    logger.info("Running pre-commit checks...")
    c.run("pre-commit run --all-files --color always", pty=True)
    c.run("safety check --full-report", warn=True, pty=True)


@task
def test(c):
    """Test this package."""
    logger.info("Running Pytest...")
    c.run(
        "env PYTHONPATH=src pytest --cov=src --cov-report term-missing --cov-report html:coverage/ tests",
        pty=True,
    )


@task
def serve(c):
    """Serve the API."""
    logger.info("Serving the API...")
    c.run(
        "uvicorn radix_se_challenge.api:app",
        pty=True,
    )


@task
def run(c):
    """Train and evaluate a model."""
    from radix_se_challenge.data import load_train_test
    from radix_se_challenge.model.usage import train, evaluate

    logger.info("Loading the data...")
    df_train, df_test = load_train_test()

    logger.info("Training the model...")
    train(df_train)

    logger.info("Evaluating the model...")
    _ = evaluate(df_test)
