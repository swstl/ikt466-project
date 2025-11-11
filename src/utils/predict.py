import torch
from utils.audio import extract_centered_clip

def predict(model, preprocessor, audio_path, threshold=0.02):
    dataset = model.dataset

    extracted = extract_centered_clip(audio_path, threshold)
    if extracted is None:
        print("Could not extract clip from audio.")
        exit()
    clip, _ = extracted

    spectrogram = preprocessor.audio_to_melspectrogram(clip)
    input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)

    # now predict
    with torch.no_grad():
        out = model(input_tensor)
        predicted_class = torch.argmax(out, dim=1)
        confidence = torch.softmax(out, dim=1)[0][predicted_class].item()
        final_class = dataset.classes[predicted_class]

    return final_class, confidence
