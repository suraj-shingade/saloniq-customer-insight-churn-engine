"""
Configuration module for the ML Pipeline service.

All settings are sourced from environment variables with sensible defaults
where applicable. No values are hardcoded in business logic.
"""

import os


POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "saloniq")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "saloniq")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
MODEL_PATH: str = os.environ.get("MODEL_PATH", "/app/models")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
