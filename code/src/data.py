import os
import pickle
from glob import glob
from typing import List

import yaml
import pandas as pd
from tqdm import tqdm

cfg_file = "config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def prepare_realgrab_data(label_idx: int):
    """
    Args:
        label_idx (int): Label index of the target signal to load.
            - 0: (other) intromission
            - 1: intromission before ejaculation
            (https://seita-lab.slack.com/archives/C02S29LFS3E/p1665393687721799?thread_ts=1665119360.992349&cid=C02S29LFS3E)
    Returns:
        signals (List): List of signal data
    """
    # Fetch pickle files
    signals = []
    realgrab_loc = os.path.join(
        config["data_path"]["data_root"],
        config["data_path"]["project_dir"],
        config["data_path"]["real_grab"]["dirname"],
        config["data_path"]["real_grab"]["version"],
    )
    fpaths = glob(realgrab_loc + "/*.pickle")

    # Load csv file.
    csvfile = os.path.join(
        realgrab_loc, 
        config["data_path"]["real_grab"]["csvname"]
    )
    df_label = pd.read_csv(csvfile, index_col=0)
    target_pickles = list(df_label[df_label.loc[:, "label"] == label_idx].index)
    # Open pickle files.
    for fpath in tqdm(fpaths):
        data_idx = os.path.splitext(os.path.basename(fpath))[0]
        if data_idx not in target_pickles:
            continue

        with open(fpath, "rb") as fp:
            data = pickle.load(fp)
        data = data.loc[:, ["ACh", "DA"]].values
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        signals.append(data.T) # (signal_length, 2) -> (2, signal_length).
    return signals