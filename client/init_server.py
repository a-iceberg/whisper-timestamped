import os
import requests
import logging


class Server:
    def __init__(self):
        self.gpu_url = os.environ.get(
            "WHISPER_SERVER_DEFAULT", "http://localhost:8000/transcribe"
        )
        self.temp_file_path = ""
        self.temp_file_name = ""

        logging.basicConfig(level=logging.INFO)

    def accept_feature_extractor(self, sentences, accept):
        if len(accept) > 1 and accept["text"] != "":
            accept_text = str(accept["text"])
            conf_score = []
            i = 0
            accept_start = 0
            accept_end = 0
            for result_rec in accept["segments"]:
                if i == 0:
                    accept_start = result_rec["start"]
                conf_score.append(float(result_rec["confidence"]))
                i += 1
            if i > 0:
                accept_end = result_rec["end"]
            sentences.append(
                {
                    "text": accept_text,
                    "start": accept_start,
                    "end": accept_end,
                    "confidence": sum(conf_score) / len(conf_score),
                }
            )

    def transcribation_process(
        self,
        duration,
        side,
        original_file_name,
        rec_date,
        src,
        dst,
        linkedid,
        file_size,
        queue_date,
        transcribation_date,
    ):
        logger_text = " size: " + str(file_size)
        logger_text += " file: " + self.temp_file_path + self.temp_file_name

        logging.info(logger_text)

        sentences = []

        file_path = self.temp_file_path + self.temp_file_name
        with open(file_path, "rb") as audio_file:
            response = requests.post(
                self.gpu_url,
                files={"file": (original_file_name, audio_file, "audio/wav")},
            )

        if response.status_code == 200:
            accept = response.json()
            self.accept_feature_extractor(sentences, accept)
        else:
            logging.error(f"Error in file processing: {response.text}")
            return 0, [], []

        for i in range(0, len(sentences)):
            self.save_result(
                duration,
                sentences[i]["text"],
                sentences[i]["start"],
                sentences[i]["end"],
                side,
                transcribation_date,
                str(sentences[i]["confidence"]),
                original_file_name,
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
        duration,
        accept_text,
        accept_start,
        accept_end,
        side,
        transcribation_date,
        conf_mid,
        original_file_name,
        rec_date,
        src,
        dst,
        linkedid,
        file_size,
        queue_date,
    ):
        logging.info("save result start")
        print("=== save_result", accept_text)
