import whisper_timestamped as whisper
import json
import os
import torch
import time
import wave
from parse import convert_json_to_text
from statistics import median


def get_audio_duration(wav_filename):
    with wave.open(wav_filename, "r") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = whisper.load_model("large-v3", device=device, download_root="./cache")
    print("Model loaded successfully")

    performance_ratios = []

    # Iterate files in data
    for filename in sorted(os.listdir("input")):
        print(f"Transcribing {filename}")

        audio_path = f"input/{filename}"
        audio = whisper.load_audio(audio_path)
        start_time = time.time()
        result = whisper.transcribe(model, audio, vad=True, language="ru")
        duration = time.time() - start_time

        with open(f"output/{filename}.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        audio_duration = get_audio_duration(audio_path)
        performance_ratio = duration / audio_duration
        performance_ratios.append(performance_ratio)

        print(
            f"Transcribed {filename} in {duration} seconds with performance ratio: {performance_ratio}"
        )

    median_performance_ratio = median(performance_ratios)
    print(f"Median Performance Ratio: {median_performance_ratio}")

    convert_json_to_text()


if __name__ == "__main__":
    main()
