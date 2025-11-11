import librosa
import numpy as np
import os
import sys
from typing import Tuple, List, Optional
import soundfile as sf
from tqdm import tqdm
import kagglehub

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.dataset import download_dataset



class AudioPreprocessor:
    """
    Preprocessor for converting audio files to spectrograms for speech command recognition.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 target_length: int = 16000):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio files
            n_mels: Number of mel frequency bins
            n_fft: Length of the FFT window
            hop_length: Number of samples between successive frames
            target_length: Target length of audio in samples (1 second at 16kHz)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
    
    def load_and_preprocess_audio(self, file_path: str) -> np.ndarray:
        """
        Load an audio file and preprocess it to a fixed length.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Preprocessed audio array
        """
        # Load audio file
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Pad or truncate to target length
        if len(audio) < self.target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        elif len(audio) > self.target_length:
            # Truncate if too long
            audio = audio[:self.target_length]
        
        return audio
    
    def audio_to_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel-scale spectrogram.
        
        Args:
            audio: Audio array
            
        Returns:
            Mel spectrogram array
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def audio_to_stft_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to STFT spectrogram.
        
        Args:
            audio: Audio array
            
        Returns:
            STFT spectrogram array (magnitude)
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Get magnitude and convert to dB
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return magnitude_db
    
    def audio_to_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mfcc = mfcc.T
        
        return mfcc
    
    def process_file(self, file_path: str, spectrogram_type: str = 'mel') -> np.ndarray:
        """
        Process a single audio file to spectrogram.
        
        Args:
            file_path: Path to the audio file
            spectrogram_type: Type of spectrogram ('mel' or 'stft')
            
        Returns:
            Spectrogram array
        """
        # Load and preprocess audio
        audio = self.load_and_preprocess_audio(file_path)
        
        # Convert to spectrogram
        if spectrogram_type == 'mel':
            return self.audio_to_melspectrogram(audio)
        elif spectrogram_type == 'stft':
            return self.audio_to_stft_spectrogram(audio)
        else:
            raise ValueError(f"Unknown spectrogram type: {spectrogram_type}")
    
    def process_dataset(self, 
                       dataset_path: str, 
                       output_dir: Optional[str] = None,
                       spectrogram_type: str = 'mel',
                       file_extensions: List[str] = ['.wav', '.flac', '.mp3'],
                       preserve_structure: bool = True,
                       skip_existing: bool = True) -> dict:
        """
        Process entire dataset directory to spectrograms while preserving folder structure.
        
        Args:
            dataset_path: Path to the dataset directory
            output_dir: Directory to save processed spectrograms (maintains same structure as input)
            spectrogram_type: Type of spectrogram ('mel' or 'stft')
            file_extensions: List of audio file extensions to process
            preserve_structure: If True and output_dir is specified, maintains exact folder structure
            skip_existing: If True, skip files that are already processed
            
        Returns:
            Dictionary with class labels as keys and lists of spectrograms as values
        """
        spectrograms = {}
        
        # Get all subdirectories (class folders)
        class_folders = [f for f in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, f))]
        
        print(f"Found {len(class_folders)} classes: {class_folders}")
        
        if output_dir and preserve_structure:
            print(f"Preserving folder structure in: {output_dir}")
        
        for class_name in tqdm(class_folders, desc="Processing classes"):
            class_path = os.path.join(dataset_path, class_name)
            spectrograms[class_name] = []
            
            # Create corresponding output directory if specified
            if output_dir:
                output_class_dir = os.path.join(output_dir, class_name)
                os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all audio files in the class folder
            audio_files = []
            for ext in file_extensions:
                audio_files.extend([f for f in os.listdir(class_path) 
                                  if f.lower().endswith(ext)])
            
            print(f"Processing {len(audio_files)} files in class '{class_name}'")
            
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}", leave=False):
                file_path = os.path.join(class_path, audio_file)
                
                if output_dir and skip_existing:
                    output_file = os.path.join(output_dir, class_name, 
                                             f"{os.path.splitext(audio_file)[0]}.npy")
                    if os.path.exists(output_file):
                        continue
                
                try:
                    # Process the audio file
                    spectrogram = self.process_file(file_path, spectrogram_type)
                    spectrograms[class_name].append(spectrogram)
                    
                    # Save to file maintaining folder structure
                    if output_dir:
                        output_file = os.path.join(output_dir, class_name, 
                                                 f"{os.path.splitext(audio_file)[0]}.npy")
                        np.save(output_file, spectrogram)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        if output_dir:
            print(f"Processed spectrograms saved to: {output_dir}")
            print("Folder structure preserved - each class has its own subdirectory")
        
        return spectrograms
    
    def process_and_save_dataset(self, dataset_path: str, output_dir: str, 
                               spectrogram_type: str = 'mel') -> None:
        """
        Process dataset and save spectrograms while maintaining exact folder structure.
        
        Args:
            dataset_path: Path to the input dataset directory
            output_dir: Path to output directory (will mirror input structure)
            spectrogram_type: Type of spectrogram ('mel' or 'stft')
        """
        print(f"Processing dataset from: {dataset_path}")
        print(f"Saving spectrograms to: {output_dir}")
        print(f"Spectrogram type: {spectrogram_type}")
        
        # Process and save maintaining structure
        self.process_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            spectrogram_type=spectrogram_type,
            preserve_structure=True
        )
        
        print("Dataset processing complete!")
        print(f"Structure preserved: {output_dir}/[class_name]/[filename].npy")
    
    def get_default_dataset_path(self) -> str:
        """
        Get the default dataset path from project's data/raw/ directory.
        Downloads dataset if not present.
        
        Returns:
            Path to the dataset in data/raw/
        """
        # Get the project root (assumes this file is in src/data/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        dataset_path = os.path.join(project_root, "data", "raw")
        
        # Check if dataset exists, if not download it
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print("Dataset not found in project. Downloading...")
            return download_dataset()
        
        print(f"Found existing dataset in: {dataset_path}")
        return dataset_path
    
    def get_default_output_dir(self) -> str:
        """
        Get the recommended output directory for spectrograms within the project.
        
        Returns:
            Path to data/processed/spectrograms/
        """
        # Get the project root (assumes this file is in src/data/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, "data", "processed", "spectrograms2")
    
    def process_dataset_mfcc(self, 
                             dataset_path: str, 
                             output_dir: str,
                             n_mfcc: int = 13,
                             file_extensions: List[str] = ['.wav', '.flac', '.mp3'],
                             skip_existing: bool = True) -> None:
        class_folders = [f for f in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, f))]
        
        print(f"Found {len(class_folders)} classes")
        
        for class_name in tqdm(class_folders, desc="Processing classes"):
            class_path = os.path.join(dataset_path, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all audio files
            audio_files = []
            for ext in file_extensions:
                audio_files.extend([f for f in os.listdir(class_path) 
                                  if f.lower().endswith(ext)])
            
            processed_count = 0
            skipped_count = 0
            
            for audio_file in tqdm(audio_files, desc=f"{class_name}", leave=False):
                output_file = os.path.join(output_class_dir, 
                                         f"{os.path.splitext(audio_file)[0]}.npy")
                
                # Check if already processed
                if skip_existing and os.path.exists(output_file):
                    skipped_count += 1
                    continue
                
                try:
                    file_path = os.path.join(class_path, audio_file)
                    
                    # Load and extract MFCC
                    audio = self.load_and_preprocess_audio(file_path)
                    mfcc = self.audio_to_mfcc(audio, n_mfcc=n_mfcc)
                    
                    # Save
                    np.save(output_file, mfcc)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error: {file_path}: {e}")
            
            if skipped_count > 0:
                print(f"  {class_name}: Processed {processed_count}, Skipped {skipped_count} existing files")
        
        print(f"✓ MFCC features saved to: {output_dir}")
    
    def process_dataset_mfcc_auto(self, 
                                  dataset_path: Optional[str] = None,
                                  n_mfcc: int = 13) -> None:

        if dataset_path is None:
            print("Using default dataset location...")
            dataset_path = self.get_default_dataset_path()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        output_dir = os.path.join(project_root, "data", "processed", "mfcc")
        
        print(f"Dataset path: {dataset_path}")
        print(f"Output path: {output_dir}")
        
        self.process_dataset_mfcc(
            dataset_path=dataset_path,
            output_dir=output_dir,
            n_mfcc=n_mfcc
        )
    
    def process_dataset_auto(self, 
                            dataset_path: Optional[str] = None,
                            output_dir: Optional[str] = None,
                            spectrogram_type: str = 'mel',
                            file_extensions: List[str] = ['.wav', '.flac', '.mp3'],
                            preserve_structure: bool = True,
                            skip_existing: bool = True) -> dict:
        """
        Process dataset with automatic path detection and recommended output location.
        
        Args:
            dataset_path: Path to dataset (if None, uses KaggleHub default)
            output_dir: Output directory (if None, uses recommended project location)
            spectrogram_type: Type of spectrogram ('mel' or 'stft')
            file_extensions: List of audio file extensions to process
            preserve_structure: If True, maintains exact folder structure
            
        Returns:
            Dictionary with class labels as keys and lists of spectrograms as values
        """
        # Use default paths if not provided
        if dataset_path is None:
            print("Using default KaggleHub dataset location...")
            dataset_path = self.get_default_dataset_path()
            
        if output_dir is None:
            print("Using recommended project spectrograms directory...")
            output_dir = self.get_default_output_dir()
        
        print(f"Dataset path: {dataset_path}")
        print(f"Output path: {output_dir}")
        
        # Process dataset
        return self.process_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            spectrogram_type=spectrogram_type,
            file_extensions=file_extensions,
            preserve_structure=preserve_structure,
            skip_existing=skip_existing
        )

def create_preprocessor(config: dict = None) -> AudioPreprocessor:
    """
    Factory function to create an AudioPreprocessor with custom configuration.
    
    Args:
        config: Dictionary with preprocessor parameters
        
    Returns:
        AudioPreprocessor instance
    """
    if config is None:
        config = {}
    
    return AudioPreprocessor(**config)

def process_audio():
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        target_length=16000
    )

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

    return preprocessor 


# Example usage
if __name__ == "__main__":
    process_audio()
