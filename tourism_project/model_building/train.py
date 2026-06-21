
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Set MLFLOW_ALLOW_FILE_STORE environment variable to true to opt out of the file store exception
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# 1. Set MLflow tracking URI and experiment name
# For local tracking, you might use 'file:./mlruns'. For remote, set the server URI.
# For this MLOps pipeline, we will assume MLflow will be set up to log to a local file system first.
# When integrating with GitHub Actions, this would typically point to a remote MLflow server.
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Wellness-Tourism-Package-Prediction-Experiment")

# Set your Hugging Face Token
os.environ["HF_TOKEN"] = "" # Removed hardcoded token
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Define Hugging Face dataset repository ID
HF_DATASET_REPO_ID = "sdncountry/tourism-project-dataset"

# Define paths for loading data from Hugging Face
Xtrain_path = f"hf://datasets/{HF_DATASET_REPO_ID}/split_data/Xtrain.csv"
Xtest_path = f"hf://datasets/{HF_DATASET_REPO_ID}/split_data/Xtest.csv"
ytrain_path = f"hf://datasets/{HF_DATASET_REPO_ID}/split_data/ytrain.csv"
ytest_path = f"hf://datasets/{HF_DATASET_REPO_ID}/split_data/ytest.csv"

print(f"Loading data from: {Xtrain_path}")

# 4. Load the dataset
try:
    X_train = pd.read_csv(Xtrain_path)
    y_train = pd.read_csv(ytrain_path).squeeze() # .squeeze() to convert DataFrame to Series
    X_test = pd.read_csv(Xtest_path)
    y_test = pd.read_csv(ytest_path).squeeze()
    print("Train and test data loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error loading data from Hugging Face: {e}")
    exit()

# Drop 'Unnamed: 0' and 'CustomerID' columns if they exist and are not features
if 'Unnamed: 0' in X_train.columns: X_train = X_train.drop('Unnamed: 0', axis=1)
if 'CustomerID' in X_train.columns: X_train = X_train.drop('CustomerID', axis=1)
if 'Unnamed: 0' in X_test.columns: X_test = X_test.drop('Unnamed: 0', axis=1)
if 'CustomerID' in X_test.columns: X_test = X_test.drop('CustomerID', axis=1)


# 6. Separate numerical and categorical variables
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

# 7. Preprocess the data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 8. Initialize a base model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# 9. Create a hyperparameter for fine tuning
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__max_depth': [3, 5],
    'xgb__subsample': [0.7, 1.0]
}

# 10. Use model pipeline to preprocess the data and build the model using the base model above
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', xgb_model)
])

# Start MLflow run
with mlflow.start_run():
    print("MLflow Run started...")
    # 11. Initialize the Grid Search CV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)

    # 12. Build the model using Grid Search CV
    grid_search.fit(X_train, y_train)
    print("Grid Search CV fitting complete.")

    # 13. Log the parameters on MLflow
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)
    print(f"Logged best parameters: {best_params}")

    # 14. Get the best model
    best_model = grid_search.best_estimator_

    # 15. Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # 16. Evaluate the performance
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # 17. Log the performance metrics on MLflow
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    })
    print("Logged evaluation metrics.")

    # 18. Save the model
    model_filename = "best_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Model saved locally as {model_filename}")

    # Define trusted types for skops.io to prevent MlflowException
    skops_trusted_types = [
        'numpy.dtype',
        'sklearn.compose._column_transformer._RemainderColsList',
        'xgboost.core.Booster',
        'xgboost.sklearn.XGBClassifier',
        'sklearn.preprocessing._data.StandardScaler',
        'sklearn.preprocessing._encoders.OneHotEncoder',
        'sklearn.impute._base.SimpleImputer',
        'sklearn.pipeline.Pipeline',
        'sklearn.compose._column_transformer.ColumnTransformer'
    ]

    # Log the model with MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="sklearn_model",
        registered_model_name="XGBoostWellnessPackagePredictor",
        skops_trusted_types=skops_trusted_types
    )
    print("Model logged to MLflow.")

    # 19. Define Hugging Face model repository ID
    HF_MODEL_REPO_ID = "sdncountry/tourism-package-model"
    HF_MODEL_REPO_TYPE = "model"

    # 20. Check if the space exists, if not, create it
    try:
        api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type=HF_MODEL_REPO_TYPE)
        print(f"Hugging Face model repository '{HF_MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Hugging Face model repository '{HF_MODEL_REPO_ID}' not found. Creating new repository...")
        create_repo(repo_id=HF_MODEL_REPO_ID, repo_type=HF_MODEL_REPO_TYPE, private=False)
        print(f"Hugging Face model repository '{HF_MODEL_REPO_ID}' created.")
    except HfHubHTTPError as e:
        print(f"Hugging Face API Error: {e}")
        print("Please ensure your HF_TOKEN is valid and has write access.")
        exit()

    # 21. Upload the saved model file to the Hugging Face model repository
    try:
        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=model_filename,
            repo_id=HF_MODEL_REPO_ID,
            repo_type=HF_MODEL_REPO_TYPE,
        )
        print(f"Model '{model_filename}' uploaded to Hugging Face model repository: {HF_MODEL_REPO_ID}")
    except HfHubHTTPError as e:
        print(f"Hugging Face API Error during model upload: {e}")
        print("Please check your HF_TOKEN and ensure the repository ID is correct.")

print("Train.py script execution complete.")
