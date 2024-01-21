"""API endpoints of the hiring challenge."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi.applications import FastAPI

app = FastAPI(
    title="SE challenge",
    description="Hiring challenge for the Software Engineer applicants.",
    version="0.0.0",
    docs_url="/",  # Put docs under default URL
)

# The API should be created here


