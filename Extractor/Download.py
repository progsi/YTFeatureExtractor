
import logging
import yt_dlp


def download(yt_id: str, outpath: str):
    """
    downloads the YouTube video with the defined ID as mp3
    :param yt_id: YouTube ID to download video for
    :return:
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outpath,
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

