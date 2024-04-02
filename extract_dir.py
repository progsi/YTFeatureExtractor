import os
from tqdm import tqdm
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from Extractor.Helper import process_file, FEAT_KEYS


def main():
    args = parse_args()
    input_dir = args.input
    parallel = args.parallel
    feat_keys = FEAT_KEYS

    input_paths = []
    output_paths = []
    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".mp3"):
                input_paths.append(os.path.join(root, name))
                output_paths.append(to_output_path(root, name))
    if parallel:
        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(process_file, input_paths, output_paths, feat_keys), total=len(input_paths)))
    else:
        for input_path, output_path in input_paths, output_paths:
            process_file(input_path, output_path, feat_keys)


def to_output_path(root: str, name: str):
    root_out = os.path.dirname(root)
    root_out = os.path.join(root_out, 'audio_features')
    name_out = name.replace(".mp3", ".h5")
    return os.path.join(root_out, name_out)

def parse_args():
    parser = argparse.ArgumentParser(description='Audio feature extractor from mp3 dir.')
    parser.add_argument('-i', '--input', type=str, default='/data/audio_data/',
                    help='Path with mp3s.')
    parser.add_argument('--parallel', action="store_true", 
                        help='Use multiple cores for extraction and downloads.')
    args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":
    main()
    

    