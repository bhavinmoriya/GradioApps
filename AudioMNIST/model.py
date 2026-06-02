import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Configuration ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Ensure dataset is downloaded and extracted (Fix for FileNotFoundError) ---
dataset_url = "https://github.com/soerenab/AudioMNIST/archive/refs/heads/master.zip"
zip_file_name = "master.zip"
extracted_folder = "AudioMNIST-master"
data_folder = os.path.join(extracted_folder, "data")

if not os.path.exists(extracted_folder):
    print(f"Downloading {zip_file_name}...")
    !wget {dataset_url} -O {zip_file_name}
    print(f"Extracting {zip_file_name}...")
    !unzip -q {zip_file_name}
    print("Extraction complete.")
else:
    print(f"Dataset already extracted at {extracted_folder}.")

if not os.path.exists(data_folder):
    print(f"Error: Data folder not found at {data_folder} even after extraction attempt.")
    # Handle error or exit if data is critically missing

# Hyperparameters
BATCH_SIZE = 16*4
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
NUM_CLASSES = 10 # Digits 0-9
SAMPLE_RATE = 16000 # Common sample rate for audio
N_MELS = 64 # Number of Mel bands to generate
TARGET_TIME_FRAMES = 81

# Dataset paths (from previous cell)
extracted_folder = "AudioMNIST-master"
data_folder = os.path.join(extracted_folder, "data")

# --- 2. AudioMNIST Dataset ---

class AudioMNISTDataset(Dataset):
    def __init__(self, data_folder, sample_rate=SAMPLE_RATE, n_mels=N_MELS, split='train'):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.audio_files = []
        self.labels = []
        self._load_audio_files()

        # Simple train-test split based on speaker IDs for now
        # AudioMNIST has 60 speakers (00-59). Let's use 00-49 for train, 50-59 for test
        train_speakers = [str(i).zfill(2) for i in range(50)]
        test_speakers = [str(i).zfill(2) for i in range(50, 60)]

        if split == 'train':
            self.filtered_files = [(f, l) for f, l in zip(self.audio_files, self.labels) if os.path.basename(os.path.dirname(f)) in train_speakers]
        elif split == 'test':
            self.filtered_files = [(f, l) for f, l in zip(self.audio_files, self.labels) if os.path.basename(os.path.dirname(f)) in test_speakers]
        else:
            self.filtered_files = list(zip(self.audio_files, self.labels))

        print(f"Loaded {len(self.filtered_files)} files for {split} split.")

    def _load_audio_files(self):
        for speaker_id_folder in os.listdir(self.data_folder):
            speaker_path = os.path.join(self.data_folder, speaker_id_folder)
            if os.path.isdir(speaker_path):
                for filename in os.listdir(speaker_path):
                    if filename.endswith('.wav'):
                        # Filename format: digit_speakerid_instance.wav (e.g., 9_25_42.wav)
                        try:
                            label = int(filename.split('_')[0])
                            self.audio_files.append(os.path.join(speaker_path, filename))
                            self.labels.append(label)
                        except ValueError:
                            # Skip files that don't match the expected naming convention
                            continue

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        audio_path, label = self.filtered_files[idx]
        waveform, sr = torchaudio.load(audio_path)

        # Resample if sample rate is different
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Apply MelSpectrogram transform
        mel_spectrogram = self.transform(waveform)

        # Ensure consistent shape (padding/truncating if necessary)
        # A simple approach for now: pad to a fixed length if too short, or truncate if too long.
        # Determine a target sequence length. Let's use the max length observed in dummy data, or a reasonable constant.
        # For AudioMNIST, audio files are short, around 1 second, which translates to ~32-81 frames at 16kHz with default settings.
        # Let's set a target length based on the typical output shape of the previous dummy data, which was 81 frames.
        TARGET_TIME_FRAMES = 81 # This was example_melspec.shape[3] from dummy data

        if mel_spectrogram.shape[2] < TARGET_TIME_FRAMES:
            # Pad with zeros if shorter
            padding = TARGET_TIME_FRAMES - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        elif mel_spectrogram.shape[2] > TARGET_TIME_FRAMES:
            # Truncate if longer
            mel_spectrogram = mel_spectrogram[:, :, :TARGET_TIME_FRAMES]

        # The model expects input of shape (batch_size, 1, n_mels, time_frames)
        # The mel_spectrogram from torchaudio.transforms is (channels, n_mels, time_frames). If mono, channels=1.
        # This is already in the expected format.

        return mel_spectrogram, label


# Create real AudioMNIST datasets
train_dataset = AudioMNISTDataset(data_folder=data_folder, split='train')
test_dataset = AudioMNISTDataset(data_folder=data_folder, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Example: check a batch's shape
example_melspec, example_label = next(iter(train_loader))
print(f"Mel-spectrogram batch shape: {example_melspec.shape}")
print(f"Labels batch shape: {example_label.shape}")

# Plot an example Mel-spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(example_melspec[0, 0].log2().numpy(), aspect='auto', origin='lower', cmap='viridis')
plt.title(f"Example Mel-Spectrogram (Label: {example_label[0].item()})")
plt.xlabel("Time Frame")
plt.ylabel("Mel Filter Bank")
plt.colorbar(format='%+2.0f dB')
plt.show()

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
        self._recalc_cnn_output_size(example_melspec.shape[2], example_melspec.shape[3])

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

model = AudioClassifier(NUM_CLASSES).to(device)
print(model)

# --- 4. Training ---

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'====> Epoch: {epoch} Average loss: {running_loss / len(train_loader.dataset):.4f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

print("\n--- Starting Training ---")
for epoch in range(1, NUM_EPOCHS + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

print("--- Training Complete ---")

# Save the model

model_path = "audio_classifier_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
