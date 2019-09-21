import pandas as pd
import os
from collections import Counter
from typing import Dict
import ipdb
import json
from tqdm import tqdm


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_table(path, names=['caption', 'url'])


def get_url_to_index(df: pd.DataFrame, split: str) -> Dict[str, str]:

    cache_path = f'data/url_to_index_{split}.json'

    if os.path.exists(cache_path):
        print('cache exists')
        with open(cache_path, 'r') as f:
            data = json.load(f)
            return data

    counter = Counter(df.url)

    res = {}
    for index, key in enumerate(tqdm(counter)):
        res[key] = f'{index:07d}'

    with open(cache_path, 'w') as f:
        json.dump(res, f)

    return res


def tsv_to_aria2_input_file(df: pd.DataFrame, split: str):
    res = []

    url_to_index = get_url_to_index(df, split)

    with open(split + '.txt', 'w') as f:
        for url, index in url_to_index.items():
            cmd = f'{url}\n  out={index}\n'
            f.write(cmd)


def main():
    df_val = read_tsv('data/Validation_GCC-1.1.0-Validation.tsv')
    tsv_to_aria2_input_file(df_val, 'val')

    df_train = read_tsv('data/Train_GCC-training.tsv')
    tsv_to_aria2_input_file(df_train, 'train')


if __name__ == "__main__":
    main()
