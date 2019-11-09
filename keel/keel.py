import os
import io
import re

import numpy as np
import pandas as pd

from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder

from typing import NamedTuple

STORAGE_DIR = os.path.join(os.path.dirname(__file__), ".storage")


def find_datasets():
    for dirpath, _, filenames in os.walk(STORAGE_DIR):
        dat_file = f"{os.path.basename(dirpath)}.dat"
        if dat_file in filenames:
            yield os.path.relpath(dirpath, STORAGE_DIR)


# TODO: Add downloading datasets
KEEL_DATASETS = tuple(find_datasets())


def parse_keel_dat(dat_file):
    with open(dat_file, "r") as fp:
        data = fp.read()
        header, payload = data.split("@data\n")

    attributes = re.findall(r"@[Aa]ttribute (.*?)[ {](integer|real|.*)", header)
    output = re.findall(r"@[Oo]utput[s]? (.*)", header)

    dtype_map = {"integer": np.int, "real": np.float}

    columns, types = zip(*attributes)
    types = [*map(lambda _: dtype_map.get(_, np.object), types)]
    dtype = dict(zip(columns, types))

    data = pd.read_csv(io.StringIO(payload), names=columns, dtype=dtype)
    target = data[output]
    data.drop(labels=output, axis=1, inplace=True)

    return data, target


def load_dataset(dataset_name, return_X_y=False):
    if dataset_name not in KEEL_DATASETS:
        raise ValueError(f"Can't find dataste {dataset_name}.")

    data_file = os.path.join(STORAGE_DIR, dataset_name, f"{os.path.basename(dataset_name)}.dat")
    data, target = parse_keel_dat(data_file)

    class_encoder = LabelEncoder()
    target = class_encoder.fit_transform(target.values.ravel())
    target_names = class_encoder.classes_

    if return_X_y:
        return data.values, target
    else:
        return Bunch(
            **{
                "data": data,
                "target": target,
                "target_names": target_names,
                "filename": data_file,
            }
        )


if __name__ == "__main__":
    for _ in KEEL_DATASETS:
        data = load_dataset(_)
        print(_)
        # print(data)
