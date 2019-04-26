import sys; sys.path.append('..')
import pandas as pd
import os
import shutil
from train.params import args
from tqdm import tqdm

if __name__ == '__main__':

    labels_path = args.labels
    train_path = args.train_data
    test_path = args.test_data
    all_files_path = args.raw_data

    labels_df = pd.read_csv(labels_path)

    train_files = labels_df['image_name'].tolist()
    all_files = os.listdir(all_files_path)
    test_files = set(set(all_files) - set(train_files))

    for _file in tqdm(train_files):
        src = os.path.join(all_files_path, _file)
        dst = os.path.join(train_path, _file)
        shutil.copy(src=src, dst=dst)

    for _file in tqdm(test_files):
        src = os.path.join(all_files_path, _file)
        dst = os.path.join(test_path, _file)
        shutil.copy(src=src, dst=dst)