# Whisper-jax Inference Server

An OpenAI Audio Speech-to-Text API-compatible inference service using Whisper-JAX and FastAPI.

## Hardware requirement
NVIDIA graphics cards supporting CUDA 12 with at least 10GB of VRAM

##　Software dependency
docker nvidia-container-toolkit
[container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## How to build
‘‘‘bash
docker build . -t whisper-jax-infer
’’’

## How to run
‘‘‘bash
docker run -d -p 8050 --runtime=nvidia --gpus all whisper-jax-infer
’’’

