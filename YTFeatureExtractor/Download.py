
import logging
import yt_dlp


def download(yt_id: str, outpath: str):
    """Downloads video identified by yt_id into output path.
    Args:
        yt_id (str): youtube identifier
        outpath (str): output path
    Returns:
        bool: flag indicating successful download
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outpath.replace(".mp3", ""),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'

        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.download([f'https://www.youtube.com/watch?v={yt_id}'])
    except yt_dlp.utils.YoutubeDLError:

        logging.error(f'{yt_id} could not be downloaded')
        return False
