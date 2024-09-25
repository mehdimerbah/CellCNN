import pandas as pd

# Path to the original NK_markers.csv
original_file = 'NK_cell_dataset/NK_markers.csv'

# Read the single row with all feature names
df = pd.read_csv(original_file, header=None)

# Transpose and save to a new file with one feature per line
df_transposed = df.transpose()
df_transposed.to_csv('NK_cell_dataset/NK_markers_corrected.csv', header=False, index=False)

print("Created NK_markers_corrected.csv with one feature per line.")
