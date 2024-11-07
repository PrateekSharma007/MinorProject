import whisper
from pydub import AudioSegment
import os
import wave
from pathlib import Path

def check_audio_content(file_path):
    """
    Check if the file contains audio content
    Returns True if file has audio, False otherwise
    """
    try:
        audio = AudioSegment.from_file(file_path)
        # Check if the audio has any samples
        if len(audio) > 0 and audio.rms > 0:
            return True
        return False
    except Exception as e:
        print(f"Error checking audio content for {file_path}: {e}")
        return False

def transcribe_audio(file_path, model):
    """
    Transcribe audio file using Whisper
    Returns transcribed text or None if transcription fails
    """
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return None

def process_audio_directory(directory_path):
    """
    Process all .webm files in the directory
    """
    # Load Whisper model once
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Create output directory for text files
    output_dir = os.path.join(directory_path, "transcriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .webm files
    webm_files = list(Path(directory_path).glob("*.webm"))
    
    print(f"Found {len(webm_files)} .webm files")
    
    # Process each file
    for file_path in webm_files:
        print(f"\nProcessing: {file_path.name}")
        
        # Check if file has audio content
        if check_audio_content(str(file_path)):
            print("Audio content detected, transcribing...")
            
            # Transcribe the audio
            transcription = transcribe_audio(str(file_path), model)
            
            if transcription:
                # Create output text file
                output_file = os.path.join(output_dir, f"{file_path.stem}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Transcription saved to: {output_file}")
            else:
                print("Transcription failed")
        else:
            print("No audio content detected")

if __name__ == "__main__":
    # Replace with your directory path
    directory_path = r"C:\Users\sharm\MinorProject\backend\audio_chunks"
    
    try:
        process_audio_directory(directory_path)
        print("\nProcessing complete!")
    except Exception as e:
        print(f"An error occurred: {e}")