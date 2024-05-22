import traceback
import threading

from kui.asgi import (
    HTTPException,
    Kui,
    OpenAPI,
    Body,
    HttpView,
    JSONResponse,
)

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from kui.wsgi.routing import MultimethodRoutes

from loguru import logger
from argparse import ArgumentParser

from http import HTTPStatus

routes = MultimethodRoutes(base_class=HttpView)


class InvokeRequest(BaseModel):
    text: str = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游."
    reference_text: Optional[str] = None
    reference_audio: Optional[str] = None
    max_new_tokens: int = 0
    chunk_length: Annotated[int, Field(ge=0, le=500, strict=True)] = 150
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.5
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    speaker: Optional[str] = None
    format: Literal["wav", "mp3", "flac"] = "wav"
    streaming: bool = False


@routes.http.post("/v1/audio/transcriptions")
def api_infer(
    req: Annotated[InvokeRequest, Body(exclusive=True)],
):
    """
    Invoke model and generate audio
    """
    # instantiate pipeline


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/text2semantic-sft-medium-v1-4k.pth",
    )
    parser.add_argument(
        "--llama-config-name", type=str, default="dual_ar_2_codebook_medium"
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/vq-gan-group-fsq-2x1024.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="vqgan_pretrain")
    parser.add_argument("--tokenizer", type=str, default="fishaudio/fish-speech-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8000")

    return parser.parse_args()


# Define utils for web server
def http_execption_handler(exc: HTTPException):
    return JSONResponse(
        dict(
            statusCode=exc.status_code,
            message=exc.content,
            error=HTTPStatus(exc.status_code).phrase,
        ),
        exc.status_code,
        exc.headers,
    )


def other_exception_handler(exc: "Exception"):
    traceback.print_exc()

    status = HTTPStatus.INTERNAL_SERVER_ERROR
    return JSONResponse(
        dict(statusCode=status, message=str(exc), error=status.phrase),
        status,
    )


def main():
    from zibai import create_bind_socket, serve

    args = parse_args()

    # Define Kui app
    openapi = OpenAPI(
        {
            "title": "Whisper Inference API",
        },
    ).routes

    app = Kui(
        routes=routes + openapi[1:],  # Remove the default route
        exception_handlers={
            HTTPException: http_execption_handler,
            Exception: other_exception_handler,
        },
        cors_config={},
    )

    logger.info(f"Warming up done, starting server at http://{args.listen}")
    sock = create_bind_socket(args.listen)
    sock.listen()

    # Start server
    serve(
        app=app,
        bind_sockets=[sock],
        max_workers=10,
        graceful_exit=threading.Event(),
    )


if __name__ == "__main__":
    main()
