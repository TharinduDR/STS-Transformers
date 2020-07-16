import csv
import glob
import os

import pandas as pd


def concatenate(path):
    files = glob.glob(
        os.path.join(path, "*.txt"))  # advisable to use os.path.join as this makes concatenation OS independent

    df_from_each_file = (pd.read_csv(f, sep='\t', header=None, names=["id", "labels", "text_a", "text_b"],
                                     quoting=csv.QUOTE_NONE) for f in files)
    df = pd.concat(df_from_each_file, ignore_index=True)

    return df


def read_test(path):
    df = pd.read_csv(os.path.join(path, "STS.input.track1.ar-ar.txt"), sep='\t', header=None, names=["text_a", "text_b"],
                                     quoting=csv.QUOTE_NONE)
    with open(os.path.join(path, "STS.gs.track1.ar-ar.txt")) as f:
        gs = f.read().splitlines()

    gs = list(map(float, gs))
    df["labels"] = gs

    return df
