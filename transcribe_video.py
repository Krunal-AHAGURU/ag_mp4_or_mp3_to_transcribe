"""
github link for modal : https://github.com/SYSTRAN/faster-whisper

Model options in faster-whisper:
Model size	Description
tiny	Very fast, low accuracy
base	Fast, medium-low accuracy
small	Medium speed, good accuracy
medium	Slower, higher accuracy
large	Slowest, best accuracy


currently we are using small 

pip install faster-whisper
pip install ffmpeg-python
pip install pandas
"""

from faster_whisper import WhisperModel
import os

def transcribe_video_to_vtt(video_path, output_vtt, model_size="small"):
    """
    Transcribes an MP4 video file into a .vtt subtitle file using faster-whisper.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Loading model '{model_size}'...")
    # model = WhisperModel(model_size, device="cuda" if model_size != "tiny" else "cpu")
    model = WhisperModel(model_size, device="cpu")

    print(f"Transcribing video: {video_path}")
    segments, info = model.transcribe(video_path)  # Directly pass video file

    print(f"Detected language: {info.language}, Probability: {info.language_probability:.2f}")

    with open(output_vtt, "w", encoding="utf-8") as vtt:
        vtt.write("WEBVTT\n\n")

        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            vtt.write(f"{i}\n")
            vtt.write(f"{start} --> {end}\n")
            vtt.write(f"{segment.text.strip()}\n\n")

    print(f"✅ Subtitles saved as {output_vtt}")

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to VTT timestamp format (HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

# Example usage:

# transcribe_video_to_vtt("lecture_video.mp4", "lecture_subtitles.vtt", model_size="small")
if __name__ == "__main__":
    video_file = "BS1064NEET26MAY18V1.mp4"  # Your input file
    output_vtt = "BS1064NEET26MAY18V1.vtt"  # Your output subtitle file

    transcribe_video_to_vtt(video_file, output_vtt, model_size="small")

