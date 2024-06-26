FROM nvidia/cuda:12.2.2-runtime-rockylinux9

RUN dnf install -y --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm
RUN dnf install -y --nogpgcheck https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm
RUN dnf -y update && dnf install -y python39 python3.9-pip gcc gcc-c++ git ffmpeg ffmpeg-devel && dnf clean all 
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /work
COPY . .

RUN pip install --no-cache-dir git+https://github.com/sanchit-gandhi/whisper-jax.git 
RUN pip install --no-cache-dir -e .

EXPOSE 8050
VOLUME [ "/work/data" ]

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8050"]
