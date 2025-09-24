import os
from pathlib import Path


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") in {"1", "true", "yes", "on"}
    TESTING = False
    INSTANCE_PATH = os.environ.get(
        "FLASK_INSTANCE_PATH",
        str(Path(__file__).resolve().parent / "instance"),
    )
