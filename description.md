# Model Training Parameters and Optimization Approach

### Command:
```bash
python main.py --ncell 500 \
               --nsubset 1000 \
               --nfeatures 36 \
               --train_data train_data.csv \
               --test_data test_data.csv \
               --output_path ./output \
               --response_data NK_cell_dataset/NK_fcs_samples_with_labels.csv \
               --response label \
               --sample_col sample_id \
               --name NK_cell_analysis \
               --feature_names NK_cell_dataset/NK_markers_corrected.csv
```
### Parameters:

| Parameter                                         | Description                                                                                           |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `--ncell 500`                                     | The number of cells per multi-cell input sample. The model processes groups of 500 cells at a time, aggregating the information from these cells for training and prediction. |
| `--nsubset 1000`                                  | The number of multi-cell input subsets generated for training. In this case, 1000 subsets of cells will be created per class or per sample, depending on the setup. |
| `--nfeatures 36`                                  | The number of features (or markers) used for each cell. The model will extract and use 36 features from the dataset. |
| `--train_data train_data.csv`                     | The path to the training data CSV file. This file contains the cell-level data used for training the model. |
| `--test_data test_data.csv`                       | The path to the test data CSV file. The test data is used to evaluate the performance of the model on unseen samples. |
| `--output_path ./output`                          | The directory where the model outputs (trained model, logs, predictions) will be saved.                  |
| `--response_data NK_cell_dataset/NK_fcs_samples_with_labels.csv` | The path to the CSV file containing the response labels or phenotypes. This file links the sample IDs to the target labels (0 or 1). |
| `--response label`                                | Specifies the column in the `response_data` file that contains the labels (e.g., disease status, binary classification). |
| `--sample_col sample_id`                          | The column name in the training and test data that contains the sample IDs. This is used to link each sample's data to its corresponding label. |
| `--name NK_cell_analysis`                         | A name for the current analysis. This is used to label the output files and logs for easy identification of this experiment. |
| `--feature_names NK_cell_dataset/NK_markers_corrected.csv` | The path to a file containing the names of the features (markers) to be used in the model. Each row in this file corresponds to a feature name. |
