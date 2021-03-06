import os
import tarfile
import urllib

import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_csv_data(path=HOUSING_PATH, filename="housing.csv"):
    file_path = os.path.join(path, filename)
    return pd.read_csv(file_path)

def display_scores(scores):
    print("scores: {}".format(scores))
    print("mean: {}".format(scores.mean()))
    print("standad deviation: {}".format(scores.std()))
