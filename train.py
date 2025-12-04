import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from model import TimeMoEWithKAN

def merge_radiomics_data(df):
    # Create a dictionary to store merged data
    merged_data = []
    grouped = df.groupby(['PID', 'Rib'])
    
    for (pid, rib), group in grouped:
        merged_row = {'PID': pid, 'Rib': rib, 'Fracture': group['Fracture'].iloc[0]}
        # Merge features from Time 1, 2, 3
        for time in [1, 2, 3]:
            time_group = group[group['Time'] == time]
            if not time_group.empty:
                for col in time_group.columns:
                    if col not in ['PID', 'Time', 'Rib', 'Fracture']:
                        merged_row[f"{col}_{time}"] = time_group[col].values[0]
        
        merged_data.append(merged_row)
    merged_df = pd.DataFrame(merged_data)
    return merged_df

# Define feature groups based on renamed feature names
def aggregate_knowledge_embeddings(features, feature_to_embedding, unique_groups):
    relevant_embeddings = [feature_to_embedding[feature] for feature in features if feature in feature_to_embedding]
    if relevant_embeddings:
        aggregated_embedding = np.mean(np.stack(relevant_embeddings), axis=0)
    else:
        aggregated_embedding = np.zeros(len(unique_groups))
    return aggregated_embedding

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Load Training Data
    data_path = "scaled_train.xlsx"
    df = pd.read_excel(data_path)
    merged_df = merge_radiomics_data(df)
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    X = merged_df.drop(columns=['PID', 'Rib', 'Fracture'])
    y = merged_df['Fracture']

    # Load Testing Data
    test_data_path = "scaled_test.xlsx"
    df_test = pd.read_excel(test_data_path)
    merged_df_test = merge_radiomics_data(df_test)
    merged_df_test = merged_df_test.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = merged_df_test.drop(columns=['PID', 'Rib', 'Fracture'])
    y_test = merged_df_test['Fracture']

    num_radiomics = 1051
    X_1 = X.iloc[:, :num_radiomics]
    X_2 = X.iloc[:, num_radiomics:2*num_radiomics]
    X_3 = X.iloc[:, 2*num_radiomics:]


    # Define feature groups based on renamed feature names
    feature_groups = {
        'Shape': [col for col in X.columns if 'shape' in col.lower()],
        'FirstOrder': [col for col in X.columns if 'firstorder' in col.lower()],
        'GLCM': [col for col in X.columns if 'glcm' in col.lower()],
        'GLRLM': [col for col in X.columns if 'glrlm' in col.lower()],
        'GLSZM': [col for col in X.columns if 'glszm' in col.lower()],
        'GLDM': [col for col in X.columns if 'gldm' in col.lower()],
        'NGTDM': [col for col in X.columns if 'ngtdm' in col.lower()],
        'Original': [col for col in X.columns if 'original' in col.lower()],
        'log3': [col for col in X.columns if 'log-sigma-3' in col.lower()],
        'log5': [col for col in X.columns if 'log-sigma-5' in col.lower()],
    }

    # Assign a one-hot encoding to each group
    unique_groups = list(feature_groups.keys())
    group_to_onehot = {group: np.eye(len(unique_groups))[i] for i, group in enumerate(unique_groups)}

    # Create knowledge embeddings for each feature
    feature_to_embedding = {}
    for group, features in feature_groups.items():
        for feature in features:
            feature_to_embedding[feature] = group_to_onehot[group]

    # Aggregate knowledge embeddings for each timestamp
    aggregated_embedding_1 = aggregate_knowledge_embeddings(X_1.columns, feature_to_embedding, unique_groups)
    aggregated_embedding_2 = aggregate_knowledge_embeddings(X_2.columns, feature_to_embedding, unique_groups)
    aggregated_embedding_3 = aggregate_knowledge_embeddings(X_3.columns, feature_to_embedding, unique_groups)

    # Step 6: Instantiate and Train the Model
    knowledge_1 = np.tile(aggregated_embedding_1, (X_1.shape[0], 1))
    knowledge_2 = np.tile(aggregated_embedding_2, (X_2.shape[0], 1))
    knowledge_3 = np.tile(aggregated_embedding_3, (X_3.shape[0], 1))

    X_train = [X_1, X_2, X_3, knowledge_1, knowledge_2, knowledge_3]

    #validation
    X_test_1 = X_test.iloc[:, :num_radiomics].rename(columns=lambda x: x[:-2])
    X_test_2 = X_test.iloc[:, num_radiomics:2*num_radiomics].rename(columns=lambda x: x[:-2])
    X_test_3 = X_test.iloc[:, 2*num_radiomics:].rename(columns=lambda x: x[:-2])

    # Generate Knowledge Embeddings for test data
    knowledge_test_1 = np.tile(aggregated_embedding_1, (X_test_1.shape[0], 1))
    knowledge_test_2 = np.tile(aggregated_embedding_2, (X_test_2.shape[0], 1))
    knowledge_test_3 = np.tile(aggregated_embedding_3, (X_test_3.shape[0], 1))

    X_test = [X_test_1, X_test_2, X_test_3, knowledge_test_1, knowledge_test_2, knowledge_test_3]

    class_weights = {1: 0.88}

    time_moe_kan = TimeMoEWithKAN(num_radiomics=num_radiomics, knowledge_dim=len(unique_groups))
    history = time_moe_kan.train(X_train, y, X_train, y, epochs=48, batch_size=48, class_weight=class_weights)
    

