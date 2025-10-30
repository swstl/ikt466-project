import kagglehub
import os

# Download the dataset from Kaggle
path = kagglehub.dataset_download("neehakurelli/google-speech-commands")

print("😡😡😡😡😡 Dataset downloaded to: 😡😡😡😡", path)

# list folders in the path folder:
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
print("😡😡😡😡 Folders in the dataset: 😡😡😡😡", folders)   


