########################################
#########        Imports       #########
########################################
from data.preprocessing import AudioPreprocessor
from data.dataset import download_dataset 
from data.dataset import create_loaders, SpectroDataset, MFCCDataset
from utils.train import train
from models import create_model 
from torchinfo import summary
from utils.predict import predict


import shutil
import torch
import os



##########################################
######### First install the data #########
##########################################
data_path = '../data/raw'
if os.path.exists(data_path):
    print(f"Dataset already exists at {data_path}. Skipping download.")
else:
    path = download_dataset()
    # move downloaded dataset to ../data/google-speech-commands
    shutil.move(path, data_path) 





###################################################
######### Preprocecss the downloaded data #########
###################################################
preprocess_path = '../data/processed'
preprocessor = AudioPreprocessor(
    sample_rate=16000,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    target_length=16000
)

if os.path.exists(preprocess_path):
    print(f"Preprocessed data already exists at {preprocess_path}. Skipping preprocessing.")
else:

    print("="*60)
    print("STEP 1: Processing spectrograms (for CNN)")
    print("="*60)
    spectrograms = preprocessor.process_dataset_auto(
        spectrogram_type='mel',
        skip_existing=True
    )

    total_files = sum(len(specs) for specs in spectrograms.values())
    print(f"\nSpectrograms processed: {total_files} files")
    print(f"Classes: {list(spectrograms.keys())}")

    print("\n" + "="*60)
    print("STEP 2: Processing MFCC features (for LSTM)")
    print("="*60)
    preprocessor.process_dataset_mfcc_auto(n_mfcc=13)

    print("\n" + "="*60)
    print("✓ All preprocessing complete!")
    print("="*60)
    print("Spectrograms saved to: data/processed/spectrograms/")
    print("MFCC features saved to: data/processed/mfcc/")
    print("="*60)






######################################
#########  Train the models  #########
######################################

train_loader, test_loader, dataset, shape = create_loaders(
    dataset_class=SpectroDataset,
    # dataset_class=MFCCDataset,
    batch_size=256,
    test_split=0.2,
    shuffle=True,
    random_seed=41,

    # does nothing on cpu:
    num_workers=40,
    pin_memory=True,
    persistent_workers=True
)

model = create_model("cnn.CNN_Wide",
    # CNN
    input_channels=shape[0],
    num_classes=len(dataset.classes),
    H=shape[1],
    W=shape[2]

    # others
    # input_size=shape[1],
    # hidden_size=128,
    # num_layers=2,
    # num_classes=len(dataset.classes),
    # dropout=0.3
)

# trained_model = train(
#     model,
#     train_loader,
#     test_loader,
#     epochs=1,
#     lr=0.001
# )

summary(model, input_size=(1, *shape))

del train_loader
del test_loader




############################################
#########  TESTING THE MODEL HERE  #########
############################################
# model evaluation mode: set model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('trained/CNN_20251110_122117_95.23.pth', map_location=device))
model.eval()

predicted, confidence = predict(
    model,
    preprocessor,
    dataset,
    "../data/kramsen/sheila.wav",
    threshold=0.02
)

print(f"Predicted class: {predicted}, with confidence {confidence*100:.2f}%")
