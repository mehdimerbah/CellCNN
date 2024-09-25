"""
Copyright 2022 Jan T. Schleicher
"""

import os
import sys
import errno
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from model import CellCnn
from utils import mkdir_p
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix, \
    r2_score, mean_squared_error, mean_absolute_error

import argparse


def get_args(argv=None):
    """
    Utility function for getting command line arguments
    @param argv: arguments passed on to argparse.ArgumentParser.parse_args()
    @return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncell", type=int, help="number of cells per multicell input", required=True)
    parser.add_argument("--nsubset", type=int, help="number of multicell inputs per class or sample", required=True)
    parser.add_argument("--nrun", type=int, help="number of neural network configurations to try", required=False,
                        default=100)
    parser.add_argument("--nfeatures", type=int, help="number of features/PCs/... to use", required=True)
    parser.add_argument("--start_feature", type=int, help="zero-based index of first feature", default=0,
                        required=False)
    parser.add_argument("--train_data", type=str, help="path to input CSV training data file", required=True)
    parser.add_argument("--test_data", type=str, help="path to input CSV test data file", required=True)
    parser.add_argument("--output_path", type=str, help="path to output folder", required=True)
    parser.add_argument("--response_data", type=str, help="path to CSV file containing response", required=True)
    parser.add_argument("--response", type=str, help="column name of response variable", required=True)
    parser.add_argument("--sample_col", type=str, help="column name of sample IDs", required=True)
    parser.add_argument("--name", type=str, help="name for this analysis", required=True)
    parser.add_argument("--feature_names", type=str, help="file with feature names (one per row, no index, no header)",
                        required=False, default=None)
    parser.add_argument("--regression", action="store_true", help="use regression (default: classification)")
    parser.add_argument("--class_order", type=str, help="order of classes for classification", nargs="+")
    parser.add_argument("--no_log", action="store_true", help="do not generate a log file")
    return parser.parse_args(args=argv)


def load_data(path: str) -> pd.DataFrame:
    """
    Utility function for loading training/test data from a CSV file
    @param path: str; path to CSV file
    @return: pandas DataFrame containing the data
    """
    data = pd.read_csv(path, index_col=0, sep=None, engine="python")
    data.index.names = ["cell_barcode"]
    data.reset_index(inplace=True)
    return data


def extract_samples(data: pd.DataFrame, info: pd.DataFrame, response_data: pd.DataFrame, sample_col: str,
                    feature_names: list, nfeatures: int):
    """
    Extract samples and corresponding response values from the data
    @param data: pandas DataFrame with data
    @param info: pandas DataFrame with metadata for samples from one dataset
    @param response_data: pandas DataFrame with metadata linking samples and phenotypes
    @param sample_col: name of column containing sample IDs
    @param feature_names: names of feature columns in the data
    @param nfeatures: number of features to use
    @return: samples: list of numpy arrays; responses: list of response values (integers for classification)
    """
    # extract the data
    samples, responses = [], []

    for sample_id in info[sample_col]:
        x = data[data[sample_col] == sample_id][feature_names]
        samples.append(np.asarray(x)[:, :nfeatures])
        responses.append(response_data[response_data[sample_col] == sample_id].reset_index(drop=True)
                         .loc[0, "response"])
    return samples, responses


def evaluate_predictions(predictions: np.ndarray, true_responses: list, regression: bool, data_set: str,
                         n_classes=None):
    """
    Evaluate predictions by comparing them to the true response values
    @param predictions: numpy array containing predictions
    @param true_responses: list of true response values
    @param regression: boolean specifying whether regression or classification is performed
    @param data_set: name of the dataset for which predictions are evaluated (train, test)
    @param n_classes: number of classes for classification
    @return:
    """
    if regression:
        # compare predictions and true phenotypes
        print(f"\n=== Model predictions ({data_set} data) ===\n\n", predictions)
        print("True responses:", true_responses)
        print(f"{data_set.capitalize()} R-squared: {r2_score(true_responses, predictions):.4f}")
        mse = mean_squared_error(true_responses, predictions)
        print(f"{data_set.capitalize()} MSE: {mse:.4f}")
        print(f"{data_set.capitalize()} RMSE: {np.sqrt(mse):.4f}")
        print(f"{data_set.capitalize()} MAE: {mean_absolute_error(true_responses, predictions):.4f}")
    else:  # classification
        # predictions contain predicted class probabilities
        predicted_classes = predictions.argmax(axis=1)

        # compare predictions and true phenotypes
        print(f"\n=== Model predictions ({data_set} data) ===\n\n", predictions)
        print("Predicted classes:", predicted_classes.tolist())
        print("True phenotypes:", true_responses)
        print(f"{data_set.capitalize()} accuracy: {accuracy_score(true_responses, predicted_classes):.4f}")
        print(f"Balanced {data_set} accuracy: {balanced_accuracy_score(true_responses, predicted_classes):.4f}")
        if n_classes > 2:
            print(f"{data_set} ROC AUC: {roc_auc_score(true_responses, predictions, multi_class='ovo')}")
        else:
            print(f"{data_set} ROC AUC: {roc_auc_score(true_responses, predictions[:, 1])}")
        print(f"{data_set.capitalize()} confusion matrix:\n", confusion_matrix(true_responses, predicted_classes))

import traceback

def main():
    print("Starting main function")  # Add this line
    try:
        args = get_args()

        # set options from command line arguments
        ncell = args.ncell
        nsubset = args.nsubset
        name = args.name
        nfeatures = args.nfeatures
        start_feature = args.start_feature
        class_order = args.class_order
        regression = args.regression
        nrun = args.nrun
        path = args.output_path

        # create log file
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
        if not args.no_log:
            old_stdout = sys.stdout
            log_dir = os.path.join(path, "run_log")
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            log_file = open(os.path.join(log_dir, "log_" + str(current_time) + ".txt"), "w+")
            print(f"Output will be redirected to {os.path.join(log_dir, 'log_' + str(current_time) + '.txt')}")
            sys.stdout = log_file

        # print command line arguments
        print(f"DICSIT model training and evaluation")
        print(f"Analysis name: {name}")
        print(f"\n=== Parameters ===\n")
        print(f"- training data: {args.train_data}")
        print(f"- test data: {args.test_data}")
        print(f"- output path: {path}")
        print(f"- response variable: {args.response}")
        print(f"- sample ID column: {args.sample_col}")
        print(f"- feature names file: {args.feature_names}")
        print(f"- regression: {regression}\n")

        # define output directory
        out_dir = os.path.join(path, "results", "out_" + str(current_time) + "_" + name + "_" + str(nfeatures))
        mkdir_p(out_dir)
        print(f"Output directory: {out_dir}")

        # load the data
        print("Loading train data...")
        train_data = load_data(args.train_data)
        print("Train data columns:", train_data.columns.tolist())
        print("Train data shape:", train_data.shape)
        print("Loading test data...")
        test_data = load_data(args.test_data)
        print("Test data columns:", test_data.columns.tolist())
        print("Test data shape:", test_data.shape)
        print("Loading response data...")
        response_data = pd.read_csv(args.response_data, sep=None)
        print("Response data columns:", response_data.columns.tolist())
        print("Response data shape:", response_data.shape)

        # set feature names
        if args.feature_names is not None:
            # Use pandas to read the feature names correctly
            feature_names_df = pd.read_csv(args.feature_names, header=None)
            # Flatten the DataFrame to a list
            feature_names = feature_names_df.iloc[:, 0].tolist()
        else:
            feature_names = [f"PC_{i + 1}" for i in range(start_feature, nfeatures + start_feature)]
        print("Feature names:", feature_names)

        # Rename the column from 'fcs_filename' to 'sample_id'
        response_data.rename(columns={'fcs_filename': 'sample_id'}, inplace=True)

        # Remove the '.fcs' extension from 'sample_id'
        response_data['sample_id'] = response_data['sample_id'].str.replace('.fcs', '', regex=False)
        print("Updated response_data sample_id by removing '.fcs' extension.")

        # load sample ids and corresponding labels
        sample_col = args.sample_col
        response_col = args.response
        print(f"Sample column: {sample_col}")
        print(f"Response column: {response_col}")

        print("Merging train data with response data...")
        patient_info_train = train_data[[sample_col]].merge(response_data, on=sample_col, how="left") \
            .drop_duplicates(subset=[sample_col, response_col]).reset_index(drop=True)
        print("Merged Train Data:")
        print(patient_info_train.head())

        print("Merging test data with response data...")
        patient_info_test = test_data[[sample_col]].merge(response_data, on=sample_col, how="left") \
            .drop_duplicates(subset=[sample_col, response_col]).reset_index(drop=True)
        print("Merged Test Data:")
        print(patient_info_test.head())

    except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            if not args.no_log:
                sys.stdout = old_stdout
            sys.exit(1)

    if not args.no_log:
        sys.stdout = old_stdout

    # shuffle the data for each sample
    np.random.seed(42)
    train_data = train_data.groupby(sample_col).apply(pd.DataFrame.sample, frac=1, random_state=42).reset_index(drop=True)
    test_data = test_data.groupby(sample_col).apply(pd.DataFrame.sample, frac=1, random_state=42).reset_index(drop=True)

    mkdir_p(os.path.join(out_dir, "model"))
    pickle.dump(train_data[["cell_barcode", sample_col]],
                open(os.path.join(out_dir, "model", "shuffled_train_data_info.p"), "wb"))
    pickle.dump(test_data[["cell_barcode", sample_col]],
                open(os.path.join(out_dir, "model", "shuffled_test_data_info.p"), "wb"))

    print(f"Train samples: {', '.join(patient_info_train[sample_col].astype(str))}")
    print(f"Test samples: {', '.join(patient_info_test[sample_col].astype(str))}")

    # convert the class labels into integers
    response_data = response_data[(response_data[sample_col].isin(patient_info_train[sample_col])) |
                                  (response_data[sample_col].isin(patient_info_test[sample_col]))]
    if regression:
        response_data["response"] = response_data[response_col]
    else:
        if class_order is not None:
            class_labels = class_order
        else:
            class_labels = sorted(response_data[response_col].unique())
        class_label_map = {lab: i for i, lab in enumerate(class_labels)}
        print("Label mapping to integers: ", class_label_map)
        response_data["response"] = response_data[response_col].map(class_label_map)

        # counts for train data
        unique, counts = np.unique(response_data.loc[response_data[sample_col].isin(patient_info_train[sample_col]),
                                                     "response"], return_counts=True)
        print("Response counts train:", dict(zip(unique, counts)))

        # counts for test data
        unique, counts = np.unique(response_data.loc[response_data[sample_col].isin(patient_info_test[sample_col]),
                                                     "response"], return_counts=True)
        print("Response counts test:", dict(zip(unique, counts)))

    train_samples, train_responses = extract_samples(data=train_data, info=patient_info_train,
                                                     response_data=response_data, sample_col=sample_col,
                                                     feature_names=feature_names, nfeatures=nfeatures)
    test_samples, test_responses = extract_samples(data=test_data, info=patient_info_test,
                                                   response_data=response_data, sample_col=sample_col,
                                                   feature_names=feature_names, nfeatures=nfeatures)

    pickle.dump(train_samples, open(os.path.join(out_dir, "model", "train_samples.p"), "wb"))
    pickle.dump(train_responses, open(os.path.join(out_dir, "model", "train_response.p"), "wb"))
    pickle.dump(test_samples, open(os.path.join(out_dir, "model", "test_samples.p"), "wb"))
    pickle.dump(test_responses, open(os.path.join(out_dir, "model", "test_response.p"), "wb"))

    print(f"\nRunning CellCNN {'regression' if regression else 'classification'}...\n")

    # parameters
    print("=== CellCNN parameters: ===")
    print(f"- nfeatures: {nfeatures}")
    print(f"- ncell: {ncell}")
    print(f"- nsubset: {nsubset}")
    print(f"- nrun: {nrun}\n")

    model = CellCnn(ncell=ncell, nsubset=nsubset, nrun=nrun, verbose=0, regression=regression, per_sample=regression)

    model.fit(train_samples=train_samples, train_phenotypes=train_responses, outdir=out_dir)

    print("Saving results...")
    pickle.dump(model, open(os.path.join(out_dir, "model", "model.p"), "wb"))
    pickle.dump(model.results, open(os.path.join(out_dir, "model", "results.p"), "wb"))

    # make predictions on the test cohort
    test_pred = model.predict(test_samples)
    pickle.dump(test_pred, open(os.path.join(out_dir, "model", "test_pred.p"), "wb"))

    # make predictions on the train cohort
    train_pred = model.predict(train_samples)
    pickle.dump(train_pred, open(os.path.join(out_dir, "model", "train_pred.p"), "wb"))

    evaluate_predictions(test_pred, test_responses, regression, "test", None if regression else len(class_labels))
    evaluate_predictions(train_pred, train_responses, regression, "train", None if regression else len(class_labels))

    if not args.no_log:
        sys.stdout = old_stdout

if __name__ == "__main__":
    print("Script is running")
    main()
    print("Script finished")



