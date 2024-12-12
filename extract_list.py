import argparse
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from YTFeatureExtractor.Helper import process_file, FEAT_KEYS
from typing import List


def main():
    args = parse_args()
    listfile = args.listfile
    parallel = args.parallel
    input_dir = args.input
    force = args.force
    feat_keys = FEAT_KEYS

    yt_ids = get_yt_ids(listfile, delimiter=args.delimiter)

    extract(input_dir, yt_ids, feat_keys, parallel, force)

def extract(input_dir: str, yt_ids: List[str], feat_keys: List[str], parallel: bool, force: bool):
    """Extract features for videos represented by list of youtube identifiers
    Args:
        input_dir (str): _description_
        yt_ids (List[str]): list of youtube identifiers
        feat_keys (List[str]): list of feature keys (eg. cqt_20, cqt_ch, ...)
        parallel (bool): whether to use parallelization
        force (bool): whether to force new download and extraction, even if features are on disk
    """

    input_paths = [get_path(input_dir, yt_id) for yt_id in yt_ids]
    output_paths = [to_output_path(input_path) for input_path in input_paths]

    if parallel:
        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(process_file, input_paths, output_paths, feat_keys, force), total=len(input_paths)))
    else:
        for (input_path, output_path) in zip(input_paths, output_paths):
            process_file(input_path, output_path, feat_keys, force)

def get_yt_ids(input_path: str, delimiter: str):
    """Get list of youtube identifiers for given file path.
    Args:
        input_path (str): _description_

    Returns:
        List[str]: youtube identifiers
    """
    df = pd.read_csv(input_path, delimiter=delimiter)
    if input_path.endswith(".csv"):
        yt_ids = df["yt_id"]
    elif input_path.endswith(".parquet"):
        yt_ids = pd.read_parquet(input_path)["yt_id"]
    else:
        yt_ids = parse_textfile(input_path)
    return list(set(yt_ids))

def get_path(base_dir: str, yt_id: str, extension: str = ".mp3"):
    return os.path.join(base_dir, yt_id[:2], yt_id + extension)

def to_output_path(input_path: str):
    """Transform input to output path.
    Args:
        input_path (str): 
    Returns:
        str: output_path
    """
    dirlist = input_path.split(os.sep)
    dirlist[-3] = "audio_features"
    dirlist[-1] = dirlist[-1].replace(".mp3", ".h5")
    return os.sep.join((dirlist))

def parse_textfile(input_path: str):
    with open(input_path, "r") as txt_file:
        lines = txt_file.readlines()
    return lines

def parse_args():
    parser = argparse.ArgumentParser(description='Audio feature extractor from mp3 dir.')
    parser.add_argument('-l', '--listfile', type=str, 
                        help="Filepath to a list of YouTube IDs to extract from.")
    parser.add_argument('--delimiter', type=str, 
                        help="Delimiter in listfile.", default=";")
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