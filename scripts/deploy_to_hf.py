import os
from huggingface_hub import HfApi, create_repo

# Load from .env file if token not in environment
token = os.environ.get("HF_TOKEN")
if not token:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    break

if not token or token == "paste_your_token_here":
    raise ValueError("Edit the .env file and replace 'paste_your_token_here' with your actual HF token")

username = "lavkesh1709"
space_name = "anime-face-generator"
repo_id = f"{username}/{space_name}"

api = HfApi(token=token)

print(f"Creating Space: {repo_id} ...")
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",
    private=False,
    exist_ok=True,
    token=token,
)
print("Space created (or already exists).")

files = [
    "app.py",
    "model.py",
    "requirements.txt",
    "README.md",
    "generator.pth",
]

for f in files:
    print(f"Uploading {f} ...")
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    print(f"  {f} done.")

print(f"\nDeployment complete!")
print(f"Your app: https://huggingface.co/spaces/{repo_id}")
