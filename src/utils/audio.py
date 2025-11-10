import librosa
import numpy as np

def extract_centered_clip(wav_path, threshold=0.02, output_duration=1.0):
    """Extract 1-second clip centered on first audio above threshold."""
    audio, sr = librosa.load(wav_path, sr=16000)

    # Calculate RMS
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]

    # Find first frame above threshold
    above_threshold = np.where(rms > threshold)[0]

    if len(above_threshold) == 0:
        print("No audio found above threshold!")
        return None

    # Get peak frame and convert to samples
    peak_frame = above_threshold[np.argmax(rms[above_threshold])]
    peak_sample = librosa.frames_to_samples(peak_frame, hop_length=512)

    # Extract 1-second clip centered on peak
    half_duration = int(output_duration * sr / 2)
    start = peak_sample - half_duration
    end = peak_sample + half_duration

    # Check if we have enough audio
    required_length = int(output_duration * sr)

    if start < 0:
        # Peak too early, shift window forward
        start = 0
        end = required_length
        print(f"Peak at {peak_sample/sr:.2f}s (too early, adjusting window)")
    elif end > len(audio):
        # Peak too late, shift window backward
        end = len(audio)
        start = end - required_length
        print(f"Peak at {peak_sample/sr:.2f}s (too late, adjusting window)")
    else:
        print(f"Peak at {peak_sample/sr:.2f}s")


    if end > len(audio):
        print(f"Warning: Audio file is too short! File is only {len(audio)/sr:.2f}s")
        return None

    clip = audio[start:end]

    print(f"Clip from {start/sr:.2f}s to {end/sr:.2f}s (duration: {len(clip)/sr:.2f}s)")

    return clip, sr

