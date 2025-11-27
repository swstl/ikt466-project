########################################
#########        Imports       #########
########################################
from data.preprocessing import AudioPreprocessor, process_audio
from data.dataset import download_dataset 
from utils.train import train
from models import create_model 
from utils.predict import predict

import shutil
import torch
import os





##########################################
######### First install the data #########
##########################################
data_path = os.path.join('..', 'data', 'raw')
if os.path.exists(data_path):
    print(f"Dataset already exists at {data_path}. Skipping download.")
else:
    path = download_dataset()
    # move downloaded dataset to ../data/google-speech-commands
    shutil.move(path, data_path) 





###################################################
######### Preprocecss the downloaded data #########
###################################################
preprocess_path = os.path.join('..', 'data', 'processed')
if os.path.exists(preprocess_path):
    print(f"Preprocessed data already exists at {preprocess_path}. Skipping preprocessing.")
    preprocessor = AudioPreprocessor()
else:
    preprocessor = process_audio()





######################################
#########  Train the models  #########
######################################
model = create_model("cnn.CNN", #filename.classname
    # hidden_size=128,
    # num_layers=2,
    # dropout=0.3,
)


trained_model = train(
    model,
    epochs=50,
    lr=0.001
)




# quit()
############################################
#########  TESTING THE MODEL HERE  #########
############################################
# model evaluation mode: set model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('test_models/CNN_20251112_130409_acc:95.80.pth', map_location=device))
model.eval()

predicted, confidence = predict(
    model,
    preprocessor,
    "../data/kramsen/kramsen-sier-BED.wav",
    threshold=0.02
)

print(f"Predicted class: {predicted}, with confidence {confidence*100:.2f}%")
