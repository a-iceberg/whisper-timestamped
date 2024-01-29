FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y git
RUN pip install \
    git+https://github.com/linto-ai/whisper-timestamped.git#egg=whisper-timestamped[dev,vad_silero,vad_auditok,test] \
    -r requirements.txt

COPY transcribe.py /app/