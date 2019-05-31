import os

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def dir_content(path):
    dirs= []
    files = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                files.append(entry)
            if entry.is_dir():
                dirs.append(entry)

    return dirs, files

def read_all_files(path, encoding):
    files_content = []
    _, files = dir_content(path)
    for file in files:
        file_path = os.path.join(path, file.name)
        with open(file_path, "r", encoding=encoding) as file_handle:
            files_content.append(file_handle.read())
    return files_content
    
def lists_eq(list1, list2):
    return np.sum(np.sort(list1) == np.sort(list2)) == len(list1)

def unique_values(df):
    unique_values = []
    for column in df:
        unique_values.append(df[column].unique())
    return unique_values

def columns_with_missing_values(df):
    miss_val_columns = []
    for col in df:
        if df[col].isnull().values.any():
            miss_val_columns.append(col)
    return miss_val_columns

def confusion_matrix_df(y_train, y_train_pr, columns=[], perc=False):
    matrix = confusion_matrix(y_train, y_train_pr)
    if not columns:
        columns = list(range(len(matrix)))
    right_col = np.sum(matrix, axis=1)
    if perc:
        matrix = matrix/right_col
        right_col = right_col/right_col
    bottom_row = np.sum(matrix, axis=0)
    bottom_row = np.append(bottom_row, np.NaN)
    right_col = right_col.reshape(-1, 1)
    bottom_row = bottom_row.reshape(1, -1)
    matrix = np.concatenate((matrix, right_col), axis=1)
    matrix = np.concatenate((matrix, bottom_row), axis=0)

    return pd.DataFrame(matrix, columns=columns+["sum"], index=columns+["sum"])


def precision_recall_fscore_support_df(y_train, y_train_pr, columns=[]):
    matrix = precision_recall_fscore_support(y_train, y_train_pr)

    if not columns:
        columns = list(range(len(matrix[0])))
    index = ["precision", "recall", "fscore", "support"]

    return pd.DataFrame(matrix, columns=columns, index=index)