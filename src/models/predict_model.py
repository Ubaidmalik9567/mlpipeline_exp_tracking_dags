import pandas as pd
import sys
import pathlib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,confusion_matrix
import json
import logging
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import dagshub

dagshub.init(repo_owner='Ubaidmalik9567', repo_name='mlpipeline_exp_tracking_dags', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/mlpipeline_exp_tracking_dags.mlflow")

# Set up logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(dataset_path: str) -> tuple:
    try:
        logging.info(f"Loading dataset from {dataset_path}.")
        dataset = pd.read_csv(dataset_path)
        xtest = dataset.iloc[:, 0:-1]
        ytest = dataset.iloc[:, -1]
        logging.info("Data loaded and split successfully.")
        return xtest, ytest
    except Exception as e:
        logging.error(f"Error loading or splitting data: {e}")
        raise

def load_save_model(file_path: str):
    try:
        logging.info(f"Loading model from {file_path}.")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    try:
        logging.info("Evaluating model performance.")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info("Model evaluation completed successfully.")
        return metrics_dict ,y_pred
    
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving metrics to {file_path}/metrics.json.")
        with open(file_path + "/metrics.json", 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved successfully.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main():
    try:
        current_dir = pathlib.Path(__file__)
        home_dir = current_dir.parent.parent.parent

        path = sys.argv[1]
        save_metrics_location = home_dir.as_posix() + "/reports"
        processed_datasets_path = home_dir.as_posix() + path + "/processed_testdata.csv"
        trained_model_path = home_dir.as_posix() + "/models/model.pkl"

        x, y = load_and_split_data(processed_datasets_path)
        model = load_save_model(trained_model_path)

        metrics_dict, ypred = evaluate_model(model, x, y)
        save_metrics(metrics_dict, save_metrics_location)

        with open("params.yaml","r") as file:
            params = yaml.safe_load((file))

        with mlflow.start_run() as run:
            mlflow.set_experiment("dagsdash hub Demo")
            mlflow.log_metric('accuracy', metrics_dict['accuracy'])
            mlflow.log_metric('precision', metrics_dict['precision'])
            mlflow.log_metric('recall', metrics_dict['recall'])
            mlflow.log_metric('auc', metrics_dict['auc'])

        for param, value in params.items():
            for key, value in value.items():
                mlflow.log_param(f'{param}_{key}', value)

                
        # Create a confusion matrix plot
        plt.figure(figsize=(6, 6))
        cf_matrix = confusion_matrix(y, ypred)
        sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save the necessary artifact
        plt.savefig('confusion_matrix.png')

        mlflow.log_artifact('confusion_matrix.png')
        mlflow.log_artifact(__file__)
        
        mlflow.sklearn.log_model(model, "GradientBoostingClassifier")
        mlflow.set_tag("autor", "Ubaid ur Rehman")

        logging.info("Main function completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()