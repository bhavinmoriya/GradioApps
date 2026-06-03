import gradio as gr
import torch
import torch.nn.functional as F
# import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# --- 1. Configuration ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# from model import AudioClassifier, preprocess_audio, device

MODEL_PATH = "audio_classifier_model.pth"
MODEL_PATH = "/content/audio_classifier_model.pth"
LABELS = [str(i) for i in range(10)]
TARGET_TIME_FRAMES = 81
NUM_CLASSES = 10 # Digits 0-9
N_MELS = 64 # Number of Mel bands to generate

# Hyperparameters
# BATCH_SIZE = 16*4
# LEARNING_RATE = 0.001
# NUM_EPOCHS = 5
# NUM_CLASSES = 10 # Digits 0-9
SAMPLE_RATE = 16000 # Common sample rate for audio
# N_MELS = 64 # Number of Mel bands to generate
# TARGET_TIME_FRAMES = 81

# --- 3. Neural Network Model ---

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AudioClassifier, self).__init__()
        # Assuming input shape is (batch_size, 1, n_mels, time_frames)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # Calculate flattened features size after CNN layers
        self._recalc_cnn_output_size(N_MELS, TARGET_TIME_FRAMES)

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _recalc_cnn_output_size(self, n_mels_in, time_frames_in):
        # Pass a dummy tensor to determine the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, n_mels_in, time_frames_in)
            output_cnn = self.cnn(dummy_input)
            self.cnn_output_size = output_cnn.flatten(1).shape[1]
        print(f"Calculated CNN output flattened size: {self.cnn_output_size}")

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1) # Flatten all dimensions except batch
        x = self.fc(x)
        return x


model = AudioClassifier().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

def preprocess_audio(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS, target_time_frames=TARGET_TIME_FRAMES):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = T.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    mel = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)

    if mel.shape[2] < target_time_frames:
        mel = torch.nn.functional.pad(mel, (0, target_time_frames - mel.shape[2]))
    elif mel.shape[2] > target_time_frames:
        mel = mel[:, :, :target_time_frames]

    return mel

# def predict(audio_file):
#     if audio_file is None:
#         return "No file uploaded."

#     audio_path = audio_file
#     mel = preprocess_audio(audio_path).to(device)

#     with torch.no_grad():
#         logits = model(mel)
#         probs = F.softmax(logits, dim=1)[0]
#         pred_idx = torch.argmax(probs).item()

#     top3 = torch.topk(probs, 3)
#     result = {LABELS[i]: float(probs[i].cpu()) for i in range(len(LABELS))}
#     return f"Predicted digit: {LABELS[pred_idx]}", result

def predict(audio_file):
    if audio_file is None:
        return "No file uploaded.", {}

    mel = preprocess_audio(audio_file).to(device)
    if mel.dim() == 3:
        mel = mel.unsqueeze(0)

    with torch.no_grad():
        logits = model(mel)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    result = {LABELS[i]: float(probs[i].cpu()) for i in range(len(LABELS))}
    return f"Predicted digit: {LABELS[pred_idx]}", result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload a WAV file"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Class probabilities"),
    ],
    title="Audio Digit Classifier",
    description="Upload a spoken digit audio file and the model will predict the digit 0-9.",
)

if __name__ == "__main__":
    demo.launch(
        # server_name="0.0.0.0", server_port=7860,
                share=True
                )

    # a, res = predict("/content/audio.wav")
    # print(a,res)

