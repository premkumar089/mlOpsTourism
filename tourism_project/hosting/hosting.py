
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# 1. Initialize the HF Token. This establishes the connection with hugging face.
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# 2. Define the Hugging Face Space repository ID and type
HF_SPACE_REPO_ID = "sdncountry/tourism-package-predictor-app"  # IMPORTANT: Replace with your desired Space ID
HF_SPACE_REPO_TYPE = "space"

# 3. Check if the Hugging Face Space exists, and create it if it doesn't.
try:
    api.repo_info(repo_id=HF_SPACE_REPO_ID, repo_type=HF_SPACE_REPO_TYPE)
    print(f"Hugging Face Space '{HF_SPACE_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Hugging Face Space '{HF_SPACE_REPO_ID}' not found. Creating new space...")
    create_repo(repo_id=HF_SPACE_REPO_ID, repo_type=HF_SPACE_REPO_TYPE, private=False, space_sdk='docker')
    print(f"Hugging Face Space '{HF_SPACE_REPO_ID}' created.")
except HfHubHTTPError as e:
    print(f"Hugging Face API Error: {e}")
    print("Please ensure your HF_TOKEN is valid and has write access to create/update spaces.")
    exit()

# 4. Upload the entire `tourism_project/deployment` folder to the specified Hugging Face Space.
DEPLOYMENT_FOLDER_PATH = "tourism_project/deployment"

try:
    api.upload_folder(
        folder_path=DEPLOYMENT_FOLDER_PATH,
        repo_id=HF_SPACE_REPO_ID,
        repo_type=HF_SPACE_REPO_TYPE,
        commit_message="Update Streamlit app deployment files"
    )
    print(f"Deployment folder '{DEPLOYMENT_FOLDER_PATH}' uploaded to Hugging Face Space: {HF_SPACE_REPO_ID}")
except HfHubHTTPError as e:
    print(f"Hugging Face API Error during folder upload: {e}")
    print("Please check your HF_TOKEN and ensure the space ID is correct.")
