"""
Copyright 2022 Jan T. Schleicher
"""

import pickle
import numpy as np
import pandas as pd
import argparse
import os


def read_data(results_file: str, train_file: str, test_file: str):
    """
    Utility function for reading in the results and data files
    @param results_file: poth to CellCnn results file
    @param train_file: path to training data file
    @param test_file: path to test data file
    @return: results: CellCnn results dictionary; data: pandas DataFrame with training and test data
    """
    results = pickle.load(open(results_file, "rb"))
    train_data = pd.read_csv(train_file, index_col=0)
    test_data = pd.read_csv(test_file, index_col=0)
    train_data["data_set"] = "train"
    test_data["data_set"] = "test"
    data = pd.concat([train_data, test_data])
    return results, data


def get_filter_response(results: dict, data: pd.DataFrame, n_features: int) -> np.ndarray:
    """
    Compute the filter response for all consensus filters
    @param results: CellCnn results dictionary
    @param data: pandas DataFrame with training and test data
    @param n_features: number of features to use
    @return: numpy array with the filter response for each filter
    """
    filters = results["selected_filters"]
    x = np.asarray(data.filter(regex="PC_"))[:, :n_features]

    if results["scaler"] is not None:
        x = results['scaler'].transform(x)

    filter_response = np.zeros((x.shape[0], len(filters)))

    for i, f in enumerate(filters):
        w = f[:n_features]
        b = f[n_features]
        filter_response[:, i] = np.sum(w.reshape(1, -1) * x, axis=1) + b

    return filter_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="input CellCnn results file", required=True)
    parser.add_argument("-n", type=int, help="number of features", required=True)
    parser.add_argument("--train_data", type=str, help="train data CSV (PCs)", required=True)
    parser.add_argument("--test_data", type=str, help="test data CSV (PCs)", required=True)
    parser.add_argument("-o", type=str, help="output CSV file", required=True)
    args = parser.parse_args()

    results, data = read_data(args.i, args.train_data, args.test_data)
    filter_response = get_filter_response(results, data, args.n)
    filter_response_df = data.filter(regex="^(?!PC_)")
    filter_response_df[[f"response_filter_{i}" for i in range(filter_response.shape[1])]] = \
        pd.DataFrame(filter_response, index=filter_response_df.index)
    os.makedirs(os.path.split(args.o)[0], exist_ok=True)
    filter_response_df.to_csv(args.o)


if __name__ == "__main__":
    main()
