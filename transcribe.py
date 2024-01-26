import whisper_timestamped as whisper
import json
import os
import time
import wave
import logging
import torch
import pandas as pd
from statistics import median
from parse import convert_json_to_text


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        model = whisper.load_model("large-v3", device=device, download_root="./cache")
        logger.info("Model loaded successfully")
    except Exception as exc:
        raise exc

    performance_ratios = []
    results = []

    # Iterate files in data
    for filename in sorted(os.listdir("input")):
        audio_path = f"input/{filename}"
        audio = whisper.load_audio(audio_path)

        try:
            start_time = time.time()
            result = whisper.transcribe(
                model,
                audio,
                vad="audiotok",
                language="ru",
                remove_empty_words=True,
                beam_size=5,
                best_of=5,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            )
            duration_transcript = time.time() - start_time

            with open(f"output/{filename}.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"{e}")
            continue

        # Skip non-wav files
        if not "wav" in filename:
            logger.warning(f"Skipping {filename}")
            continue

        duration_audio = get_audio_duration(audio_path)
        performance_ratio = duration_transcript / duration_audio
        performance_ratios.append(performance_ratio)

        results.append(
            {
                "filename": filename,
                "duration_audio": duration_audio,
                "duration_transcript": duration_transcript,
                "performance_ratio": performance_ratio,
            }
        )

    convert_json_to_text()

    try:
        median_performance_ratio = median(performance_ratios)
        logger.info(f"Median Performance Ratio: {median_performance_ratio}")

        # Save results to DataFrame and export to CSV
        df = pd.DataFrame(results)
        df.to_csv("/app/output/performance_results.csv", index=False)
    except Exception as ex:
        logger.warning(
            f"Performance ratio wasn't calculated, wav files are missing. Error: {ex}"
        )


if __name__ == "__main__":
    main()
