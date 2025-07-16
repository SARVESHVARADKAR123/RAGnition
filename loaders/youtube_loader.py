import re
from youtube_transcript_api import YouTubeTranscriptApi
from langsmith.run_helpers import traceable  # ✅ Tracing enabled

@traceable(name="Extract YouTube Video ID")
def extract_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    Traced in LangSmith as 'Extract YouTube Video ID'.
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

@traceable(name="YouTube Transcript Loader")
def get_transcript_from_youtube(url):
    """
    Fetches and returns the full transcript text from a YouTube video.
    Traced in LangSmith as 'YouTube Transcript Loader'.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL"

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([segment["text"] for segment in transcript_list])
    except Exception as e:
        return f"Error fetching transcript: {e}"
