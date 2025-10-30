######################################################
######### run this first to install the data #########
######################################################

import shutil
from src.data.dataset import download_dataset 

path = download_dataset()

# move downloaded dataset to ./data/google-speech-commands
shutil.move(path, './data/raw')
