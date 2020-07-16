import glob
import os

import pandas as pd


def concatenate(path):
    files = glob.glob(
        os.path.join(path, "*.txt"))  # advisable to use os.path.join as this makes concatenation OS independent

    df_from_each_file = (pd.read_csv(f, sep='\t', header=None, names=["id", "labels", "text_a", "text_b"]) for f in
                         files)
    df = pd.concat(df_from_each_file, ignore_index=True)

    return df

