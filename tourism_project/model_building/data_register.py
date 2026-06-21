
import os
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

# Initialize HF_TOKEN from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Define the Hugging Face dataset repository ID
HF_DATASET_REPO_ID = "sdncountry/tourism-project-dataset" # IMPORTANT: Replace with your desired dataset repository ID
HF_DATASET_REPO_TYPE = "dataset"

# Check if the dataset repository exists, if not, create it
try:
    api.repo_info(repo_id=HF_DATASET_REPO_ID, repo_type=HF_DATASET_REPO_TYPE)
    print(f"Hugging Face dataset repository '{HF_DATASET_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Hugging Face dataset repository '{HF_DATASET_REPO_ID}' not found. Creating new repository...")
    create_repo(repo_id=HF_DATASET_REPO_ID, repo_type=HF_DATASET_REPO_TYPE, private=False)
    print(f"Hugging Face dataset repository '{HF_DATASET_REPO_ID}' created.")
except HfHubHTTPError as e:
    print(f"Hugging Face API Error: {e}")
    print("Please ensure your HF_TOKEN is valid and has write access to create/update repositories.")
    exit()

# Upload the local data folder to the Hugging Face dataset repository
DATA_FOLDER_PATH = "/content/drive/My Drive/GL Course/MLOps/tourism_project/data"

try:
    api.upload_folder(
        folder_path=DATA_FOLDER_PATH,
        repo_id=HF_DATASET_REPO_ID,
        repo_type=HF_DATASET_REPO_TYPE,
        commit_message="Upload initial tourism dataset"
    )
    print(f"Data from '{DATA_FOLDER_PATH}' uploaded to Hugging Face dataset repository: {HF_DATASET_REPO_ID}")
except HfHubHTTPError as e:
    print(f"Hugging Face API Error during data upload: {e}")
    print("Please check your HF_TOKEN and ensure the repository ID is correct.")
