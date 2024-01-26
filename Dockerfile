FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/linto-ai/whisper-timestamped.git#egg=whisper-timestamped[dev,vad_silero,vad_auditok,test]

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY transcribe.py parse.py /app/

CMD ["python", "transcribe.py"]