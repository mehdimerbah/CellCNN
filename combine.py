import pandas as pd
import os
import random

csv_dir = 'NK_cell_dataset/NK_cell_dataset/csv_files_NK'
train_ratio = 0.7

all_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
random.shuffle(all_files)

split_index = int(len(all_files) * train_ratio)
train_files = all_files[:split_index]
test_files = all_files[split_index:]

def combine_csv_files(file_list, output_file):
    dfs = []
    for f in file_list:
        df = pd.read_csv(os.path.join(csv_dir, f))
        sample_id = os.path.splitext(f)[0]  # Remove .csv extension
        df['sample_id'] = sample_id
        dfs.append(df)
    combined_df = pd.concat(dfs)
    combined_df.to_csv(output_file, index=False)

combine_csv_files(train_files, 'train_data.csv')
combine_csv_files(test_files, 'test_data.csv')
