import argparse
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from Extractor.Helper import process_file, FEAT_KEYS


def main():
    args = parse_args()
    listfile = args.listfile
    parallel = args.parallel
    input_dir = args.input
    force = args.force
    feat_keys = FEAT_KEYS

    yt_ids = get_yt_ids(listfile)

    extract(input_dir, yt_ids, feat_keys, parallel, force)

def extract(input_dir, yt_ids, feat_keys, parallel, force):

    input_paths = [get_path(input_dir, yt_id) for yt_id in yt_ids]
    output_paths = [to_output_path(input_path) for input_path in input_paths]

    if parallel:
        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(process_file, input_paths, output_paths, feat_keys, force), total=len(input_paths)))
    else:
        for (input_path, output_path) in zip(input_paths, output_paths):
            process_file(input_path, output_path, feat_keys, force)

def get_yt_ids(input_path):

    if input_path.endswith(".csv"):
        yt_ids = safe_parse_csv(input_path)
    elif input_path.endswith(".parquet"):
        yt_ids = pd.read_parquet(input_path)["yt_id"]
    else:
        yt_ids = parse_textfile(input_path)
    return list(set(yt_ids))

def get_path(base_dir: str, yt_id: str, extension: str = ".mp3"):
    return os.path.join(base_dir, str(ord(yt_id[0])), yt_id + extension)

def to_output_path(input_path: str):
    dirlist = input_path.split(os.sep)
    dirlist[-3] = "audio_features"
    dirlist[-1] = dirlist[-1].replace(".mp3", ".h5")
    return os.sep.join((dirlist))

def safe_parse_csv(input_path):
    
    try:
        df = pd.read_csv(input_path)
    except:
        df = pd.read_csv(input_path, sep=";")
    return df["yt_id"]

def parse_textfile(input_path):
    with open(input_path, "r") as txt_file:
        lines = txt_file.readlines()
    return lines

def parse_args():
    parser = argparse.ArgumentParser(description='Audio feature extractor from mp3 dir.')
    parser.add_argument('-l', '--listfile', type=str, 
                        help="Filepath to a list of YouTube IDs to extract from.")
    parser.add_argument('-i', '--input', type=str, default='/data/audio_data/',
                    help='Path with mp3s.')
    parser.add_argument('--parallel', action="store_true", 
                        help='Use multiple cores for extraction and downloads.')
    parser.add_argument('--force', action="store_true", 
                    help='Force new feature extraction even if file exists.')
    args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":
    main()