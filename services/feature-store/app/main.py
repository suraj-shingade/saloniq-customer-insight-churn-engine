"""FastAPI application for the SalonIQ Feature Store service.

Exposes precomputed customer features via REST endpoints, backed by
Redis for low-latency reads and PostgreSQL for durable storage.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import psycopg2
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.features import compute_all_features

logger = logging.getLogger(__name__)

_pg_conn = None
_redis_client = None
_ready = False
_features_computed = 0


def _configure_logging() -> None:
    """Set up structured logging based on the configured log level."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _connect_postgres(max_retries: int = 10, delay: float = 3.0):
    """Establish a PostgreSQL connection with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
            )
            conn.autocommit = False
            logger.info(
                "PostgreSQL connection established on attempt %d", attempt
            )
            return conn
        except psycopg2.OperationalError:
            logger.warning(
                "PostgreSQL connection attempt %d/%d failed, retrying in %.1fs",
                attempt,
                max_retries,
                delay,
            )
            if attempt == max_retries:
                raise
            time.sleep(delay)


def _connect_redis(max_retries: int = 10, delay: float = 3.0):
    """Establish a Redis connection with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True,
            )
            client.ping()
            logger.info(
                "Redis connection established on attempt %d", attempt
            )
            return client
        except redis.ConnectionError:
            logger.warning(
                "Redis connection attempt %d/%d failed, retrying in %.1fs",
                attempt,
                max_retries,
                delay,
            )
            if attempt == max_retries:
                raise
            time.sleep(delay)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan manager -- startup and shutdown logic."""
    global _pg_conn, _redis_client, _ready, _features_computed

    _configure_logging()
    logger.info("Feature store service starting")

    _pg_conn = _connect_postgres()
    _redis_client = _connect_redis()

    logger.info("Running initial feature computation")
    _features_computed = compute_all_features(_pg_conn, _redis_client)
    _ready = True
    logger.info(
        "Feature store ready -- %d features computed", _features_computed
    )

    yield

    logger.info("Feature store service shutting down")
    if _pg_conn and not _pg_conn.closed:
        _pg_conn.close()
        logger.info("PostgreSQL connection closed")
    if _redis_client:
        _redis_client.close()
        logger.info("Redis connection closed")


app = FastAPI(
    title="SalonIQ Feature Store",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint indicating service readiness."""
    if not _ready:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "features_computed": 0},
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "features_computed": _features_computed,
        },
    )


@app.get("/features/{salon_id}/{customer_id}")
async def get_features(salon_id: str, customer_id: str) -> dict[str, Any]:
    """Retrieve precomputed features for a specific customer."""
    if not _ready:
        raise HTTPException(
            status_code=503, detail="Service not yet ready"
        )

    redis_key = f"features:{salon_id}:{customer_id}"
    raw = _redis_client.get(redis_key)

    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for salon={salon_id}, customer={customer_id}",
        )

    return json.loads(raw)
