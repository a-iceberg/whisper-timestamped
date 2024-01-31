import argparse
import os
import requests


def upload_file(file_path: str):
    if not os.path.isfile(file_path):
        print("The file does not exist.")
        return

    url = "http://localhost:8080/transcribe"
    file = {"file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")}

    try:
        response = requests.post(url, files=file)
        response.raise_for_status()
        print("Response:", response.json())
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload an audio file to the FastAPI server for transcription."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="File path of the audio file"
    )

    args = parser.parse_args()
    upload_file(args.file)
