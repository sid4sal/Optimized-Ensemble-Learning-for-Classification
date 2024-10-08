import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from joblib import load
from xgboost import XGBClassifier
import tensorflow as tf

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score


def load_data(file_path):
    X_test = pd.read_csv(file_path)
    expected_columns = ['ID'] + [f'Column{i}' for i in range(22)]

    # Check if X_test has exactly 23 columns and if their names match the expected columns
    if list(X_test.columns) != expected_columns:
        raise ValueError("The dataset does not have 23 columns or the column names do not match.")

    return X_test


def preprocess(X_test, save_folder):
    # Drop the 'ID' column
    X_test.drop(columns=['ID'], inplace=True)

    # Caping/Binning the test dataset
    cap_dict = load(save_folder+'preprocessing_config/cap_dict.joblib')
    for col in cap_dict:
        #print(f'{col} : low= {cap_dict[col][0]} | high= {cap_dict[col][1]}')
        X_test[col] = np.clip(X_test[col], cap_dict[col][0], cap_dict[col][1])

    # Feature Engineering
    most_frequent_value_15 = load(save_folder+'preprocessing_config/most_frequent_value_15.joblib')
    X_test['Column9_NA'] = X_test['Column9'].isna().astype(int)
    X_test['Column14_NA'] = X_test['Column14'].isna().astype(int)
    X_test['Column15_Frequent'] = (X_test['Column15'] == most_frequent_value_15).astype(int)

    # Scaling (Handling Categorical Data - Min-Max normalization)
    multi_catogarical_columns = ['Column0', 'Column16', 'Column17']
    min_max_scaler = MinMaxScaler()
    min_max_scaler = load(save_folder+'preprocessing_config/min_max_scaler.joblib')
    X_test[multi_catogarical_columns] = min_max_scaler.transform(X_test[multi_catogarical_columns])

    # Scaling (Z-Normalization)
    non_catogarical_columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column14', 'Column15']
    z_scaler = StandardScaler()
    z_scaler = load(save_folder+'preprocessing_config/z_scaler.joblib')
    X_test[non_catogarical_columns] = z_scaler.transform(X_test[non_catogarical_columns])

    # Imputation : Handling the missing values
    mice_columns = ['Column0', 'Column3', 'Column4', 'Column5', 'Column6', 'Column8', 'Column9', 'Column14', 'Column15']
    mice_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, tol=0.01, random_state=42)
    # Load the imputer
    mice_imputer = load(save_folder+'preprocessing_config/mice_imputer.joblib')
    # Use the loaded imputer to transform new data
    X_test[mice_columns] = mice_imputer.transform(X_test[mice_columns])
    # If there are any other missing values (in the columns where training dataset didn't have any), handel them using median Imputation
    median_imputer = SimpleImputer(strategy='median')
    # Perform median imputation
    X_test = median_imputer.fit_transform(X_test)

    return X_test


def make_predictions(model_1, model_2, model_meta, X_test):
    # Extract Probabilities from Model 1 and Model 2
    pred_1_test = model_1.predict_proba(X_test)[:, 1]
    pred_2_test = model_2.predict(X_test)
    preds_test = np.column_stack((pred_2_test, pred_1_test))

    # Make the final Predictions using Meta Model
    Y_pred_prob = model_meta.predict(preds_test)
    Y_pred = (Y_pred_prob > 0.5).astype(int).flatten()

    return Y_pred


def evaluate_model(Y_test, Y_pred):
    # check if both the sets have same number rows
    if Y_test.shape[0] != Y_pred.shape[0]:
        raise ValueError("Number of true samples and predicted samples doesn't match!")

    # check if both the sets have same IDs for each row
    if not (Y_test['ID'] == Y_pred['ID']).all():
        raise ValueError("IDs do not match between true labels and predicted samples!")

    # Drop the 'ID' column
    Y_test.drop(columns=['ID'], inplace=True)
    Y_pred.drop(columns=['ID'], inplace=True)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Meta Model Accuracy: {accuracy * 100:.2f}%")
    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    # Classification report
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))
    # Precision, Recall, F1-Score
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")


def main(args):
    # Load test data
    X_test = load_data(args.test_data_input)

    ids = X_test['ID']  # Store the 'ID' column

    # Dataset Preprocessing
    X_test = preprocess(X_test, args.save_folder)
    
    # Load the model
    # Model 1 (Robust to class 1)
    model_1 = XGBClassifier()
    model_1.load_model(args.save_folder+'tf_models/model_xgb_robust_class_1.json')
    # Model 2 (Robust to class 0)
    model_2 = tf.keras.models.load_model(args.save_folder+'tf_models/model_nn_robust_class_0.keras')
    # Meta Model (Neural Network)
    model_meta = tf.keras.models.load_model(args.save_folder+'tf_models/model_meta_nn.keras')

    # Make predictions
    Y_pred = make_predictions(model_1, model_2, model_meta, X_test)
    
    # Save predictions to CSV
    Y_pred = pd.DataFrame({
        'ID': ids,
        'Predictions': Y_pred
    })
    Y_pred.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    
    # Evaluate the model if true labels are provided
    if args.true_labels:
        Y_test = pd.read_csv(args.true_labels)
        evaluate_model(Y_test, Y_pred)


def ensure_trailing_slash(path):
    if not isinstance(path, str):
        raise ValueError("The save_folder path must be a string.")
    if not path.endswith('/'):
        path += '/'
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions using the trained Ensemble model then save the predicted labels in a csv file, and optionally evaluate them using given true labels (if the true lable csv file path is given).')

    parser.add_argument('--test_data_input', type=str, required=True, help='Path to the test dataset (CSV file).')
    parser.add_argument('--save_folder', type=ensure_trailing_slash, required=True, help='Path to the saved model folder (models and preprocessing config files).')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the predictions (CSV file).')
    parser.add_argument('--true_labels', type=str, help='Optional path to the true labels (target labels) for evaluation (CSV file).')

    args = parser.parse_args()
    main(args)
