import argparse
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from YTFeatureExtractor.Helper import process_file, FEAT_KEYS
from extract_list import get_path, to_output_path


def main():
    
    args = parse_args()
    yt_id = args.youtube_id
    input_dir = args.input
    force = args.force
    feat_keys = FEAT_KEYS

    extract(input_dir, yt_id, feat_keys, force)


def extract(input_dir, yt_id, feat_keys, force):

    input_path = get_path(input_dir, yt_id)
    output_path = to_output_path(input_path)

    process_file(input_path, output_path, feat_keys, force)


def parse_args():
    parser = argparse.ArgumentParser(description='Audio feature extractor from mp3 dir.')
    parser.add_argument('-yt', '--youtube_id', type=str, 
                        help="Filepath to a list of YouTube IDs to extract from.")
    parser.add_argument('-i', '--input', type=str, default='/data/audio_data/',
                    help='Path with mp3s.')
    parser.add_argument('--force', action="store_true", 
                    help='Force new feature extraction even if file exists.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()