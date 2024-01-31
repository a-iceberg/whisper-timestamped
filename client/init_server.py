import os
import argparse
import requests
import logging


class Server:
    def __init__(self):
        self.gpu_url = os.environ.get(
            "WHISPER_SERVER_DEFAULT", "http://localhost:8080/transcribe"
        )
        # self.temp_file_path = ""
        # self.temp_file_name = ""

        logging.basicConfig(level=logging.INFO)

    def accept_feature_extractor(self, sentences, accept):
        if len(accept) > 1 and accept["text"] != "":
            for segments_rec in accept["segments"]:
                segment_text = str(segments_rec["text"])
                segment_start = segments_rec["start"]
                segment_end = segments_rec["end"]
                conf_score = float(segments_rec["confidence"])
                sentences.append(
                    {
                        "text": segment_text,
                        "start": segment_start,
                        "end": segment_end,
                        "confidence": conf_score,
                    }
                )

    def transcribation_process(
        self,
        original_file_name,
        duration=0,
        side=True,
        rec_date="31.01.2024",
        src=1,
        dst=2,
        linkedid=3,
        file_size=0,
        queue_date="31.01.2024",
        transcribation_date="31.01.2024",
    ):
        # logger_text = " size: " + str(file_size)
        # logger_text += " file: " + self.temp_file_path + self.temp_file_name

        # logging.info(logger_text)

        sentences = []

        # file_path = self.temp_file_path + self.temp_file_name

        file_path = original_file_name
        with open(file_path, "rb") as audio_file:
            response = requests.post(
                self.gpu_url,
                files={"file": (os.path.basename(file_path), audio_file, "audio/wav")},
            )

        if response.status_code == 200:
            accept = response.json()
            self.accept_feature_extractor(sentences, accept)
        else:
            logging.error(f"Error in file processing: {response.text}")
            return 0, [], []

        for i in range(0, len(sentences)):
            self.save_result(
                original_file_name,
                duration,
                sentences[i]["text"],
                sentences[i]["start"],
                sentences[i]["end"],
                side,
                transcribation_date,
                str(sentences[i]["confidence"]),
                rec_date,
                src,
                dst,
                linkedid,
                file_size,
                queue_date,
            )

        phrases = [sentences[i]["text"] for i in range(len(sentences))]
        confidences = [sentences[i]["confidence"] for i in range(len(sentences))]

        return len(sentences), phrases, confidences

    def save_result(
        self,
        original_file_name,
        duration,
        accept_text,
        accept_start,
        accept_end,
        side,
        transcribation_date,
        conf_mid,
        rec_date,
        src,
        dst,
        linkedid,
        file_size,
        queue_date,
    ):
        logging.info("save result start")
        print("=== save_result", accept_text)


def main():
    parser = argparse.ArgumentParser(
        description="Send an audio file to the FastAPI server for processing."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="File path of the audio file"
    )
    args = parser.parse_args()

    server = Server()
    num_sentences, phrases, confidences = server.transcribation_process(
        original_file_name=args.file
    )
    print(f"Processed {num_sentences} sentences.")


if __name__ == "__main__":
    main()
