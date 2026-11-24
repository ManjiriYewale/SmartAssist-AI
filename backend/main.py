"""FastAPI application entry point:  ``uvicorn backend.main:app``"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend import services
from backend.api.routes import public, router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    services.init_app()
    yield


app = FastAPI(title="SmartAssist AI API", version="1.0.0", lifespan=lifespan,
              description="Multi-tool RAG agent: documents, semantic search, chat, tools.")
app.include_router(public)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)  # details only in the log
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
