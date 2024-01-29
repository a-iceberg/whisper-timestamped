import os
import time
import wave
import json
import logging
from uuid import uuid4
from typing import List
from statistics import median

import torch
import pandas as pd
import whisper_timestamped as whisper
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

logging.basicConfig(
    filename="/app/logs/logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger()


def get_audio_duration(wav_filename):
    # Calculate the duration of an audio file
    with wave.open(wav_filename, "r") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration


async def transcribe_audio(files: List[UploadFile], request_type: str):
    results = []
    performance_ratios = []

    for file in files:
        # Checking and saving the file
        if not file.filename.endswith(".wav"):
            continue

        filename = f"{uuid4()}.wav"
        file_path = os.path.join(os.path.dirname(__file__), "input", filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Processing the audio
        try:
            start_time = time.time()
            audio = whisper.load_audio(file_path)
            result = whisper.transcribe(
                model,
                audio,
                vad="auditok",
                language="ru",
                remove_empty_words=True,
                beam_size=5,
                best_of=5,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            )
            duration_transcript = time.time() - start_time

            with open(f"/app/output/{filename}.json", "w") as json_file:
                json.dump(result, json_file, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error in processing file {file.filename}: {e}")
            continue

        # Statistic generation
        duration_audio = get_audio_duration(file_path)
        performance_ratio = duration_transcript / duration_audio
        performance_ratios.append(performance_ratio)

        results.append(
            {
                "filename": file.filename,
                "duration_audio": duration_audio,
                "duration_transcript": duration_transcript,
                "performance_ratio": performance_ratio,
                "request_type": request_type,
            }
        )

        # Deleting a file to save space on the server
        if os.path.exists(file_path):
            os.remove(file_path)

    median_performance_ratio = median(performance_ratios)
    logger.info(f"Median Performance Ratio: {median_performance_ratio}")

    # Saving the results in CSV
    df = pd.DataFrame(results)
    df.to_csv("/app/output/performance_results.csv", index=False)

    return median_performance_ratio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    model = whisper.load_model("large-v3", device=device, download_root="./cache")
    logger.info("Model loaded successfully")
except Exception as exc:
    raise exc


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.get("/")
async def main():
    # HTML-form for testing in a web browser
    html_content = """
            <body>
            <form action="/transcribe/single" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
            </form>
            </body>
            """
    return HTMLResponse(content=html_content)


@app.post("/transcribe/single")
async def transcribe_audio_single(files: List[UploadFile]):
    # Endpoint for single processing
    median_performance_ratio = await transcribe_audio(files, "single")
    return {"median_performance_ratio": median_performance_ratio}


@app.post("/transcribe/parallel")
async def transcribe_audio_parallel(files: List[UploadFile]):
    # Endpoint for parallel processing
    median_performance_ratio = await transcribe_audio(files, "parallel")
    return {"median_performance_ratio": median_performance_ratio}
