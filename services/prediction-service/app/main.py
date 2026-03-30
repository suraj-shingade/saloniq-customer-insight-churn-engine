import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router, shutdown, startup

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage application lifecycle: model loading and connection teardown."""
    logger.info("Starting prediction service...")
    await startup()
    logger.info("Prediction service ready.")
    yield
    logger.info("Shutting down prediction service...")
    await shutdown()
    logger.info("Prediction service stopped.")


app = FastAPI(
    title="SalonIQ Prediction Service",
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

app.include_router(router)


@app.get("/")
async def root():
    """Service identity endpoint."""
    return {"service": "SalonIQ Prediction Service", "version": "1.0.0"}
