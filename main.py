from typing import Annotated
from fastapi import FastAPI, Form, File
from whisper_jax import FlaxWhisperPipline

import jax.numpy as jnp

app = FastAPI()
pipeline = None


def preload(audio):
    global pipeline

    # instantiate pipeline in bfloat16
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16)
    return pipeline(audio)


@app.post("/v1/audio/transcriptions")
def api_infer(file: Annotated[bytes, File()], model: Annotated[str, Form()]):
    """
    Invoke model and recognize audio
    """
    global pipeline

    if pipeline is None:
        return preload(file)
    else:
        return pipeline(file)


@app.get("/preload")
def preload_model():
    """
    Preload model
    """
    global pipeline

    if pipeline is None:
        preload("audio.mp3")

    return {"status": "ok"}
