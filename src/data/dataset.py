import kagglehub

# Download the dataset from Kaggle
def download_dataset():
    """ Downloads the Google Speech Commands dataset from Kaggle """
    path = kagglehub.dataset_download("neehakurelli/google-speech-commands")
    print("😡😡😡😡😡 Dataset downloaded to: 😡😡😡😡", path)
    return path
