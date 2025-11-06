import os
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def find_random_sample():
    mfcc_dir = 'data/processed/mfcc'
    spec_dir = 'data/processed/spectrograms'
    
    classes = [d for d in os.listdir(mfcc_dir) if os.path.isdir(os.path.join(mfcc_dir, d))]
    random_class = random.choice(classes)
    
    mfcc_class_path = os.path.join(mfcc_dir, random_class)
    files = [f for f in os.listdir(mfcc_class_path) if f.endswith('.npy')]
    random_file = random.choice(files)
    
    mfcc_path = os.path.join(mfcc_dir, random_class, random_file)
    spec_path = os.path.join(spec_dir, random_class, random_file)
    
    mfcc = np.load(mfcc_path)
    spectrogram = np.load(spec_path)
    
    return mfcc, spectrogram, random_class, random_file

def visualize():
    mfcc, spec, class_name, filename = find_random_sample()
    
    print(f"Class: {class_name}")
    print(f"File: {filename}")
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Spectrogram shape: {spec.shape}")
    
    hop_length = 512
    sample_rate = 16000
    duration = (spec.shape[1] * hop_length) / sample_rate
    print(f"Calculated duration from spectrogram: {duration:.2f} seconds")
    print(f"Number of time frames: {spec.shape[1]}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    librosa.display.specshow(mfcc.T, x_axis='time', hop_length=hop_length, sr=sample_rate, cmap='viridis')
    plt.colorbar()
    plt.title(f'MFCC - {class_name}')
    plt.ylabel('MFCC Coefficients')
    
    plt.subplot(1, 2, 2)
    librosa.display.specshow(spec, x_axis='time', y_axis='mel', hop_length=hop_length, sr=sample_rate, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {class_name}')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize()