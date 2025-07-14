from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript_from_youtube(url):
    """
    Fetches and joins the transcript text from a YouTube video.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL"

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([segment["text"] for segment in transcript_list])
    except Exception as e:
        return f"Error fetching transcript: {e}"

