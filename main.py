from typing import Annotated
from fastapi import FastAPI, Form, File
from whisper_jax import FlaxWhisperPipline

from contextlib import asynccontextmanager
import jax.numpy as jnp
import os

context = {}


async def preload(data_file):
    pipeline = FlaxWhisperPipline(checkpoint=os.path.join("./data"), dtype=jnp.bfloat16)
    pipeline(data_file)
    context["pipeline"] = pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    await preload("audio.mp3")
    yield
    context.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
def api_infer(file: Annotated[bytes, File()], model: Annotated[str, Form()]):
    """
    Invoke model and recognize audio
    """
    pipeline = context.get("pipeline")
    if pipeline is None:
        return preload(file)
    else:
        return pipeline(file)


@app.get("/v1/health")
def api_health():
    return {"status": "ok"}
