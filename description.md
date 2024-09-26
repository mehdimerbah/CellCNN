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


### 2. Optimization Approach:

The training involves the following optimization approach:

- **Training/Validation Split**: The dataset is split into training and test sets, with a portion of the data held out for validation during training.
  
- **Subset Sampling**: A random or biased selection of subsets is generated. This is determined by `--nsubset 1000`, meaning 1000 multi-cell input subsets are generated from the dataset for training.

- **Cross-Validation**: The model likely uses cross-validation (e.g., Stratified K-Fold) to split the training data into multiple folds, optimizing across these folds.

- **Early Stopping**: Early stopping is employed, monitoring the validation loss and stopping the training if the performance does not improve over a certain number of epochs.

- **Model Selection**: Multiple models are trained using different hyperparameter combinations, and the best-performing models are selected based on the validation accuracy or loss.

- **Final Evaluation**: The model is finally evaluated on the test data using metrics such as accuracy, ROC AUC, and the confusion matrix.

### 3. Neural Network Hyperparameters:

| Hyperparameter              | Description |
|-----------------------------|-------------|
| **`ncell`**                 | The number of cells per multi-cell input (500 in this case). The model processes each multi-cell input to capture relationships between cells. |
| **`nsubset`**               | The number of multi-cell input samples generated for training. These subsets are created randomly or based on some selection criteria (1000 subsets are generated). |
| **`nfilter_choice`**        | The list of candidate numbers of filters for the convolutional layers. For example, the model may choose a filter size randomly from a predefined range (e.g., [3, 4, 5, 6]). |
| **`maxpool_percentages`**   | A list specifying the percentages of cells that will be max-pooled per filter. Pooling reduces the number of cells after applying filters (e.g., [1%, 5%, 20%, 100%]). |
| **`coeff_l1`**              | Coefficient for L1 regularization. This regularization term helps prevent overfitting by penalizing large weights in the network. |
| **`coeff_l2`**              | Coefficient for L2 regularization. Like L1, L2 regularization helps avoid overfitting by penalizing large weights. |
| **`learning_rate`**         | The learning rate for the Adam optimizer. If set to `None`, the script will try learning rates from a range (e.g., [0.001, 0.01]). |
| **`dropout`**               | Dropout probability for regularization. If set to `'auto'`, dropout is applied based on the model configuration. |
| **`dropout_p`**             | Dropout probability value (e.g., 0.5). This controls the percentage of neurons that are dropped out during training. |
| **`nrun`**                  | The number of neural network configurations to try during training (100 in this case). Each configuration uses different combinations of hyperparameters. |
| **`max_epochs`**            | Maximum number of epochs for training. Early stopping may stop training earlier if validation performance stops improving. |
| **`patience`**              | Number of epochs for early stopping. If the validation loss doesn't improve after a certain number of epochs (e.g., 5 epochs), the training stops. |
| **`regression`**            | If set to `True`, the model performs regression. If `False` (default), it performs classification. |
| **`dendrogram_cutoff`**     | Cutoff for hierarchical clustering of filter weights, used for post-training analysis of the learned filters. The cutoff value typically ranges between 0 and 


### 4. Key Model Components:

- **Convolutional Layers**: These layers apply convolutional filters to the input data, capturing spatial or temporal relationships between cells. The number of filters used is determined by the `nfilter_choice` hyperparameter.

- **Max Pooling**: Pooling layers reduce the dimensionality of the input, retaining the most important information. This is controlled by `maxpool_percentages` and `k`, which defines the number of cells pooled. The percentage of cells pooled is specified by the model during training.

- **Regularization**: L1 and L2 regularization (`coeff_l1`, `coeff_l2`) are applied to the convolutional and fully connected layers to prevent overfitting by penalizing large weights, making the model more robust.

- **Dropout**: Dropout (`dropout`, `dropout_p`) is used to randomly deactivate neurons during training, further reducing the risk of overfitting. When dropout is enabled, a certain proportion of the neurons are ignored during training, helping to generalize the model.

- **Optimizer**: The Adam optimizer is used for training, with a tunable learning rate (`learning_rate`). Adam is an adaptive learning rate optimizer that combines the advantages of both momentum and RMSProp optimizers, which speeds up convergence and helps avoid local minima.

### 5. Output:

# Model Evaluation Results

**Test accuracy:** 0.7518  
**Balanced test accuracy:** 0.8014  
**Precision:** 0.6017  
**Recall:** 1.0000  
**F1-score:** 0.7513  
**Matthews Correlation Coefficient:** 0.6022  
**Test ROC AUC:** 0.9821  

## Test Confusion Matrix:
|     | Predicted 0 | Predicted 1 |
|-----|-------------|-------------|
| **Actual 0** | 3014        | 1986        |
| **Actual 1** | 0           | 3000        |

## Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.60   | 0.75     | 5000    |
| 1     | 0.60      | 1.00   | 0.75     | 3000    |

**Overall Accuracy:** 0.75  
**Macro Average Precision:** 0.80  
**Macro Average Recall:** 0.80  
**Macro Average F1-Score:** 0.75  
**Weighted Average Precision:** 0.85  
**Weighted Average Recall:** 0.75  
**Weighted Average F1-Score:** 0.75

