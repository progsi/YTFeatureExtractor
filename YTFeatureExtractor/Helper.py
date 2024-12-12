import h5py
import os
import logging
import librosa
import numpy as np
from YTFeatureExtractor.PyCQT import PyCqt
from YTFeatureExtractor.SBBC import SBBC
from YTFeatureExtractor.Download import download
from typing import List

FEAT_KEYS = ["cqt_ch", "cqt_20", "cens", "onset_env", "melodia"]


def process_file(input_file: str, output_file: str, feat_keys: List[str], force=False):
    """Get features for audio file at input_file path and write into output file. If the input_file is not
    on disk, it gets downloaded and extracted afterwards.
    Args:
        input_file (str): input file path (mp3)
        output_file (str): output file path with extracted features (h5)
        feat_keys (List[str]): feature type keys (eg. cqt_20, cqt_ch)
        force (bool, optional): Whether to force redownload. Defaults to False.

    Returns:
        bool: successful extraction
    """
    yt_id = os.path.basename(input_file).replace(".mp3", "")

    print(f"Processing: {yt_id}")

    # if mp3 file not on disk, download it
    if not os.path.isfile(input_file) or force:
        try:
            download(yt_id, input_file)
            if not os.path.isfile(input_file):
                logging.error(f"Video {yt_id} unavailable!")
                return False
        except Exception as e:
            logging.exception(f"Video {yt_id} could not be downloaded!", str(e))
            return False
    
    # load mp3
    try:
        y, sr = librosa.load(input_file, sr=22050)
    except Exception as e:
        logging.exception(f"MP3 {yt_id}.mp3 could not be loaded with Librosa!", str(e))
        return False
    
    # extract features
    try:
        with h5py.File(output_file, "a") as file_out:
            
            for feat_key in feat_keys:
                extract_feature(y, sr, input_file, feat_key, file_out, force)
    except:
        logging.error(f"HDF file error {yt_id}")


def extract_cqt_20(y: np.array, sr: int = 22_050):
    """Extract cqt features as used in CQTNet.
    Args:
        y (np.array): Audio signal waveform
        sr (int, optional): Sampling rate. Defaults to 22_050.
    Returns:
        np.array: cqt spectogram of type cqt_20
    """

    def downsampling(cqt: np.array, mean_size: int = 20):
        cqt = np.abs(cqt)
        height, length = cqt.shape
        new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
        return new_cqt
    return downsampling(librosa.cqt(y=y, sr=sr))


def extract_cqt_ch(y: np.array, sr: int = 16_000, hop_size: float = 0.04):
    """Extract cqt features as used by CoverHunter.
    Args:
        y (np.array): Audio signal waveform
        sr (int, optional): Sampling rate. Defaults to 16_000.
        hop_size (float, optional): Hop size. Defaults to 0.04.
    Returns:
        bool: cqt spectogram of type cqt_ch
    """
    y = y / max(0.001, np.max(np.abs(y))) * 0.999
    py_cqt = PyCqt(sample_rate=sr, hop_size=hop_size)
    cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
    return cqt

def extract_feature(y: np.array, sr: int, mp3_path: str, feat_key: str, file_out: str, force: bool = False):
    """_summary_
    Args:
        y (np.array): audio signal waveform
        sr (int): sampling rate
        mp3_path (str): path to mp3
        feat_key (str): feature type key (eg. cqt_20, cqt_ch,...)
        file_out (str): output filepath
        force (bool, optional): Whether to force redownload. Defaults to False.
    """
    if force and feat_key in file_out.keys():
        del file_out[feat_key]
        print(f"Deleted {feat_key}")

    if not feat_key in file_out.keys():
        
        try:
            if feat_key not in ["melodia", "crepe"]:
                feature = __extract(y, sr, feat_key)
            else:
                feature = __extract_path(mp3_path, feat_key)
            file_out.create_dataset(feat_key, data=feature, compression='gzip')
        except Exception as e:
            logging.error(f"Exception {e} for {feat_key}")

        print(f"Extracted {feat_key} feature")

    else:
        print(f"{feat_key} feature already in file.")

def __extract_path(mp3_path: str, feat_key: List[str]):
    """Extract SBBC features.
    Args:
        mp3_path (str): mp3 file path
        feat_key (List[str]): list of feature keys
    Returns:
        np.array: Extracted features.
    """
    extractor = SBBC(melodia_algo=feat_key)
    return extractor(mp3_path)

def __extract(y: np.array, sr: int, feat_key: str):
    """_summary_
    Args:
        y (np.array): audio signal waveform
        sr (int): sampling rate
        feat_key (str): feature type key
    Returns:
        np.array: extracted features
    """
    if feat_key == "cqt_20":
        return extract_cqt_20(y)
    elif feat_key == "cqt_ch":
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return extract_cqt_ch(y)
    elif feat_key == "cens":
        return librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512)
    elif feat_key == "onset_env":
        return librosa.onset.onset_strength(y=y, sr=sr)
