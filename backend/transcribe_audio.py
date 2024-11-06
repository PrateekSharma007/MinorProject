import sys
import whisper
import json

# Ensure the .wav file path is provided
if len(sys.argv) < 2:
    print("Usage: python transcribe_audio.py <path_to_wav_file>")
    sys.exit(1)

# Load the file path from arguments
wav_file_path = sys.argv[1]

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the audio
result = model.transcribe(wav_file_path)

# Output transcription as JSON to stdout
print(json.dumps({"text": result["text"]}))
