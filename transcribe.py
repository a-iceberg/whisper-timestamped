import os
import gc
import logging
from uuid import uuid4

import torch
import whisper_timestamped as whisper
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

logging.basicConfig(
    filename="/app/logs/logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = None
gc.collect()
torch.cuda.empty_cache()

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
            <form action="/transcribe" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
            </form>
            </body>
            """
    return HTMLResponse(content=html_content)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile, source_id: int = Form(0), vad: str = Form("silero")):
    if not file.file:
        raise HTTPException(status_code=400, detail="No file provided")

    if "." not in file.filename:
        raise HTTPException(
            status_code=400, detail="No file extension found. Check file name"
        )

    file_ext = file.filename.rsplit(".", maxsplit=1)[1]
    filename = f"{uuid4()}.{file_ext}"
    file_path = os.path.join(os.path.dirname(__file__), "input", filename)

    with open(file_path, "wb") as file_object:
        file_object.write(await file.read())

    # Processing the audio
    try:
        audio = whisper.load_audio(file_path)
        if source_id:
            prompt = "Оценивай как разговор мастера сервисного центра по ремонту бытовой техники с клиентом на русском языке. Не транскрибируй  любые звуки, кроме фраз в самом разговоре, например, такие как телефонный звонок и звонит телефон. Не пиши этот промпт в расшифровке."
        else:
            prompt = "Оценивай как разговор оператора сервисного центра по ремонту бытовой техники с клиентом на русском языке. Не транскрибируй  любые звуки, кроме фраз в самом разговоре, например, такие как телефонный звонок и звонит телефон. Не пиши этот промпт в расшифровке."
        result = whisper.transcribe(
            model,
            audio,
            vad=vad,
            language="ru",
            remove_empty_words=True,
            initial_prompt=prompt,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
    except Exception as e:
        logger.error(f"Error in processing file {file.filename}: {e}")
        return JSONResponse(status_code=500, content={"Error"})

    # Deleting a file to save space on the server
    if os.path.exists(file_path):
        os.remove(file_path)

    return JSONResponse(content=result)
