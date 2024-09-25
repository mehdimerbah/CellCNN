# get train data from file 'train_data.csv'
# get test data from file 'test_data.csv'

import pandas as pd

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
response_data = pd.read_csv('NK_cell_dataset/NK_fcs_samples_with_labels.csv')



print("Train data columns:", train_data.columns)
print("Test data columns:", test_data.columns)
print("Response data columns:", response_data.columns)
