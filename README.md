## Whisper timestamped
ASR microservice developed for [call center transcription service](https://github.com/a-iceberg/stt_server)

Based on [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)

---

### Results
The latest version of the Whisper model - [v3](https://github.com/openai/whisper/discussions/1762), is used; service can operate on both GPU and CPU, but significantly slower on the latter. [Prompt engineering](https://github.com/a-iceberg/whisper-timestamped/blob/9cb99bddf801b01ee3c187d0909035f8dcaf4aa8/transcribe.py#L70) was applied to improve the transcription results.

Transcription quality on Russian ([source](https://github.com/a-iceberg/whisper_model_evaluator/blob/whisper/reports/whisper_comparator.ipynb)):
* *WER* - **0.2**
* *MER* - **0.2**
* *WIL* - **0.25**
