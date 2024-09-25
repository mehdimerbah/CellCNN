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

### Parameters:
