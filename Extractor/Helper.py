import h5py
import os
import logging
import librosa
import numpy as np
from Extractor.PyCQT import PyCqt
from Extractor.SBBC import SBBC
from Extractor.Download import download

FEAT_KEYS = ["cqt_ch", "cqt_20", "cens", "onset_env", "melodia"]


def process_file(input_file, output_file, feat_keys, force=False):
    yt_id = os.path.basename(input_file).replace(".mp3", "")

    print(f"Processing: {yt_id}")

    if not os.path.isfile(input_file) or force:
        try:
            download(yt_id, input_file)
            if not os.path.isfile(input_file):
                logging.error(f"Video {yt_id} unavailable!")
                return False
        except Exception as e:
            logging.exception(f"Video {yt_id} could not be downloaded!", str(e))
            return False
    try:
        y, sr = librosa.load(input_file, sr=22050)
    except Exception as e:
        logging.exception(f"MP3 {yt_id}.mp3 could not be loaded with Librosa!", str(e))
        return False
    
    try:
        with h5py.File(output_file, "a") as file_out:
            
            for feat_key in feat_keys:
                extract_feature(y, sr, input_file, feat_key, file_out, force)
    except:
        logging.error(f"HDF file error {yt_id}")


def extract_cqt_20(y, sr=22050):

    def downsampling(cqt, mean_size=20):
        cqt = np.abs(cqt)
        height, length = cqt.shape
        new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
        return new_cqt
    return downsampling(librosa.cqt(y=y, sr=sr))


def extract_cqt_ch(y, sr: int = 16000, hop_size=0.04):
    y = y / max(0.001, np.max(np.abs(y))) * 0.999
    py_cqt = PyCqt(sample_rate=sr, hop_size=hop_size)
    cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
    return cqt

def extract_feature(y, sr, mp3_path, feat_key: str, file_out, force: bool = False):


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


def __extract_path(mp3_path, feat_key):


    extractor = SBBC(melodia_algo=feat_key)
    return extractor(mp3_path)
    


def __extract(y, sr, feat_key: str):

    if feat_key == "cqt_20":
        return extract_cqt_20(y)
    elif feat_key == "cqt_ch":
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return extract_cqt_ch(y)
    elif feat_key == "cens":
        return librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512)
    elif feat_key == "onset_env":
        return librosa.onset.onset_strength(y=y, sr=sr)
