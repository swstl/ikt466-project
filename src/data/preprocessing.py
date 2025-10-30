import librosa
import numpy as np
import os
from typing import Tuple, List, Optional
import soundfile as sf
from tqdm import tqdm
import kagglehub
from .dataset import download_dataset_to_project

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
                       preserve_structure: bool = True) -> dict:
        """
        Process entire dataset directory to spectrograms while preserving folder structure.
        
        Args:
            dataset_path: Path to the dataset directory
            output_dir: Directory to save processed spectrograms (maintains same structure as input)
            spectrogram_type: Type of spectrogram ('mel' or 'stft')
            file_extensions: List of audio file extensions to process
            preserve_structure: If True and output_dir is specified, maintains exact folder structure
            
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
            Path to the dataset in data/raw/google-speech-commands/
        """
        # Get the project root (assumes this file is in src/data/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        dataset_path = os.path.join(project_root, "data", "raw", "google-speech-commands")
        
        # Check if dataset exists, if not download it
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print("Dataset not found in project. Downloading...")
            return download_dataset_to_project()
        
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
        return os.path.join(project_root, "data", "processed", "spectrograms")
    
    def process_dataset_auto(self, 
                            dataset_path: Optional[str] = None,
                            output_dir: Optional[str] = None,
                            spectrogram_type: str = 'mel',
                            file_extensions: List[str] = ['.wav', '.flac', '.mp3'],
                            preserve_structure: bool = True) -> dict:
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
            preserve_structure=preserve_structure
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

# Example usage
if __name__ == "__main__":
    # Example configuration for Google Speech Commands dataset
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        target_length=16000  # 1 second at 16kHz
    )
    
    # Process dataset with automatic paths
    print("Processing dataset with automatic path detection...")
    spectrograms = preprocessor.process_dataset_auto()
    
    # Print summary
    total_files = sum(len(specs) for specs in spectrograms.values())
    print(f"\nProcessing complete!")
    print(f"Classes processed: {len(spectrograms)}")
    print(f"Total files processed: {total_files}")
    print(f"Classes: {list(spectrograms.keys())}")
    
    print("\nAudioPreprocessor ready for use!")