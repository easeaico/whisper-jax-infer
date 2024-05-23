from typing import Annotated
from fastapi import FastAPI, Form, File
from whisper_jax import FlaxWhisperPipline

app = FastAPI()
pipeline = None


def preload(audio):
    global pipeline
    pipeline = FlaxWhisperPipline("openai/whisper-large-v3")
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
