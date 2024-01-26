import os
import json


def convert_json_to_text():
    data = []
    filenames = []

    # Read json files from ./output folder
    for filename in sorted(os.listdir("output")):
        # Skip non-json files
        if not "json" in filename:
            print(f"Skipping {filename}")
            continue
        print(f"Reading {filename}")
        with open(f"output/{filename}") as file:
            data.append(json.load(file))
            filenames.append(filename.replace(".json", ""))

    # Combining and sorting the segments from both JSON files by their start times
    combined_segments = data[0]["segments"] + data[1]["segments"]
    sorted_combined_segments = sorted(combined_segments, key=lambda x: x["start"])

    # Preparing the conversation text in the desired format
    conversation_text = []
    for segment in sorted_combined_segments:
        speaker = filenames[0] if segment in data[0]["segments"] else filenames[1]
        text = segment["text"]
        conversation_text.append(f"- {speaker}: {text}")

    # Joining the conversation lines into a single string
    conversation_str = "\n".join(conversation_text)

    # Output the conversation text to a file
    with open("output/transcript.txt", "w") as file:
        file.write(conversation_str)


def main():
    convert_json_to_text()


if __name__ == "__main__":
    main()
