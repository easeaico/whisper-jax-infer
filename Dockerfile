FROM nvidia/cuda:12.4.1-cudnn-devel-rockylinux9

RUN dnf -y update && dnf install -y python39 python3.9-pip git && dnf clean all 
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /work
COPY . .

RUN pip install --no-cache-dir git+https://github.com/sanchit-gandhi/whisper-jax.git 
RUN pip install --no-cache-dir -e .

EXPOSE 8050

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8050"]
