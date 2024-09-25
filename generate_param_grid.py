"""
Copyright 2022 Jan T. Schleicher
"""

from sklearn.model_selection import ParameterGrid
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name for this analysis", required=True)
    parser.add_argument("--data_path", type=str, help="path to data folder", required=True)
    parser.add_argument("--train_data_prefix", type=str, help="prefix of train data file name (before _split_{i}.csv)",
                        required=False, default="train_data")
    parser.add_argument("--test_data_prefix", type=str, help="prefix of test data file name (before _split_{i}.csv)",
                        required=False, default="test_data")
    parser.add_argument("--response_data", type=str, help="path to CSV file containing response", required=True)
    parser.add_argument("--nsplit", type=int, help="number of train-test splits", required=False, default=3)
    parser.add_argument("--output_path", type=str, help="path to output folder", required=True)
    parser.add_argument("--ncell", type=int, help="number of cells per multi-cell input", nargs="+", default=[500],
                        required=False)
    parser.add_argument("--nsubset", type=int, help="number of multi-cell inputs per sample/class", nargs="+",
                        default=[1000], required=False)
    parser.add_argument("--nfeatures_range", type=int, help="range of number of input features, as min, max, interval",
                        default=(10, 100, 10), nargs=3, required=False)
    parser.add_argument("--start_feature", help="zero-based index of first feature", default=0, required=False)
    parser.add_argument("--feature_names", type=str, help="file with feature names (one per row, no index, no header)",
                        required=False, default=None)
    parser.add_argument("--response", type=str, help="column name of response variable", required=True)
    parser.add_argument("--sample_col", type=str, help="column name of sample IDs", required=True)
    parser.add_argument("--regression", action="store_true", help="use regression (default: classification)")
    parser.add_argument("--class_order", type=str, help="order of classes for ordinal regression", nargs="+")
    args = parser.parse_args()

    param_grid = {"ncell": args.ncell,
                  "nsubset": args.nsubset,
                  "nfeatures": range(args.nfeatures_range[0], args.nfeatures_range[1]+1, args.nfeatures_range[2]),
                  "data": [(os.path.join(args.data_path, f"{args.train_data_prefix}_split_{i+1}.csv"),
                            os.path.join(args.data_path, f"{args.test_data_prefix}_split_{i+1}.csv"))
                           for i in range(args.nsplit)]}

    name = " ".join(["--name", args.name])
    response_data = " ".join(["--response_data", args.response_data])
    response = " ".join(["--response", args.response])
    sample_col = " ".join(["--sample_col", args.sample_col])
    output_path = " ".join(["--output_path", args.output_path])

    parameter_grid = ParameterGrid(param_grid)

    for params in parameter_grid:
        nc = " ".join(["--ncell", str(params["ncell"])])
        ns = " ".join(["--nsubset", str(params["nsubset"])])
        nf = " ".join(["--nfeatures", str(params["nfeatures"])])
        train_data = " ".join(["--train_data", params["data"][0]])
        test_data = " ".join(["--test_data", params["data"][1]])

        cmd = " ".join(["python3 main.py", nc, ns, nf, train_data, test_data, response_data, response,
                        output_path, sample_col, name])
        if args.feature_names is not None:
            cmd = " ".join([cmd, "--feature_names", args.feature_names])
        if args.regression:
            cmd = " ".join([cmd, "--regression"])
        if args.start_feature > 0:
            cmd = " ".join([cmd, "--start_feature", args.start_feature])
        if args.class_order is not None:
            cmd = " ".join([cmd, "--class_order", " ".join(args.class_order)])
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    main()
