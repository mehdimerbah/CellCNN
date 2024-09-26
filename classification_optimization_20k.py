import numpy as np
import pandas as pd
import os
import fcsparser
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")


# Define input and output directories
WDIR = os.getcwd()
DATA_PATH = os.path.join(WDIR, 'data')
FCS_DATA_PATH = os.path.join(DATA_PATH, 'gated_NK')
DATA_LABELS_PATH = os.path.join(DATA_PATH, 'NK_fcs_samples_with_labels.csv')
MARKERS_PATH = os.path.join(DATA_PATH, 'NK_markers.csv')
OUTPUT_PATH = os.path.join(WDIR, 'output')
PLOT_PATH = os.path.join(OUTPUT_PATH, 'plots')

# Load fcs file labels
labels = pd.read_csv(DATA_LABELS_PATH)

markers = ['CD3', 'CD27', 'CD19', 'CD4', 'CD8', 'CD57', '2DL1-S1', 'TRAIL', '2DL2-L3-S2',
           'CD16', 'CD10', '3DL1-S1', 'CD117', '2DS4', 'ILT2-CD85j', 'NKp46', 'NKG2D',
           'NKG2C', '2B4', 'CD33', 'CD11b', 'NKp30', 'CD122', '3DL1', 'NKp44', 'CD127', '2DL1',
           'CD94', 'CD34', 'CCR7', '2DL3', 'NKG2A', 'HLA-DR', '2DL4', 'CD56', '2DL5', 'CD25']
    
# Load FCS data and add labels to the data for every file from the labels file 
def load_fcs_data(fcs_folder):
    data_list = []
    for filename in os.listdir(fcs_folder):
        if filename.endswith(".fcs"):
            path = os.path.join(fcs_folder, filename)
            meta, data = fcsparser.parse(path)
            data.head()
            data['filename'] = filename
            data['label'] = labels[labels['fcs_filename'] == filename]['label'].values[0]
            data_list.append(data)
    return pd.concat(data_list)

data = load_fcs_data(FCS_DATA_PATH)



# Data subsampling
def subsample_scale_data(data, size=10000):
    subsample = data.sample(n=size, random_state=42, ignore_index=True)
    subsample_labels = subsample['label']
    markers = ['CD3', 'CD27', 'CD19', 'CD4', 'CD8', 'CD57', '2DL1-S1', 'TRAIL', '2DL2-L3-S2',
           'CD16', 'CD10', '3DL1-S1', 'CD117', '2DS4', 'ILT2-CD85j', 'NKp46', 'NKG2D',
           'NKG2C', '2B4', 'CD33', 'CD11b', 'NKp30', 'CD122', '3DL1', 'NKp44', 'CD127', '2DL1',
           'CD94', 'CD34', 'CCR7', '2DL3', 'NKG2A', 'HLA-DR', '2DL4', 'CD56', '2DL5', 'CD25']
    
    subsample = subsample[markers]

    scaler = StandardScaler()
    subsample_scaled = scaler.fit_transform(subsample)
    

    return subsample_scaled, subsample_labels


X_20k, y_20k = subsample_scale_data(data, size=20000)

def apply_pca(data, n_components=15):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    return pca_result

# Apply PCA
X_20k_pca = apply_pca(X_20k)

# Define a function for logging model performance metrics
def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted')
    }
    if y_prob is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
    
    return metrics


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_20k_pca, y_20k, test_size=0.2, random_state=42)

# Define hyperparameter grids for optimization
param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    },
    'SVM': {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
}


## Let's define our classifiers like we did before 
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'RandomForest': RandomForestClassifier()
}

# Set up optimization and evaluation
optimized_results = {}

for name, clf in classifiers.items():
    print(f"Optimizing {name}...")
    if name == 'LogisticRegression' or name == 'SVM':
        # Use GridSearchCV for Logistic Regression and SVM
        search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    else:
        # Use RandomizedSearchCV for Random Forest
        search = RandomizedSearchCV(clf, param_distributions=param_grids[name], cv=5, n_iter=10, scoring='accuracy', n_jobs=-1)
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    #y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
    y_prob = None
    
    # save the best model
    joblib.dump(best_model, f"{name}_model.pkl")

    ## save the prediction results
    np.save(f"{name}_y_pred.npy", y_pred)

    # Log the evaluation metrics
    print(y_prob)
    metrics = evaluate_model(y_test, y_pred, y_prob)
    optimized_results[name] = {
        'Best Params': search.best_params_,
        'Metrics': metrics
    }
    
    print(f"{name} Best Params: {search.best_params_}")
    print(f"{name} Performance:\n", pd.DataFrame(metrics, index=[0]))