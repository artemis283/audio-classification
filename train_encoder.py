import soundata
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
from models import AudioEncoder
import torch.nn as nn

dataset = soundata.initialize('urbansound8k')

# loading the .csv file
metadata = pd.read_csv('/Users/artemiswebster/sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')

class UrbanSoundDataset(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata
        self.sample_rate = 22050
        self.n_mels = 80
        self.n_time = 3000

    def __len__(self):
        return len(self.metadata)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = f'/Users/artemiswebster/sound_datasets/urbansound8k/audio/fold{row["fold"]}/{row["slice_file_name"]}'
        
        # Load and preprocess audio
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        
        # Pad audio if it's too short
        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)))
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) * 2 - 1
        
        # Pad or truncate to fixed length
        if mel_spec.shape[1] < self.n_time:
            pad_width = self.n_time - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
        else:
            mel_spec = mel_spec[:, :self.n_time]
        
        # Convert to tensor - shape will be (80, 3000)
        mel_spec = torch.FloatTensor(mel_spec)
        
        # Get label
        label = row['classID']
        
        return mel_spec, label

def get_fold_data(metadata, folds):
    return metadata[metadata['fold'].isin(folds)]

train_dataset = get_fold_data(metadata, folds=range(1, 10))
test_dataset = get_fold_data(metadata, folds=[10])

train_dataset = UrbanSoundDataset(metadata=train_dataset)
test_dataset = UrbanSoundDataset(metadata=test_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def main():
    torch.manual_seed(42)

    learning_rate = 0.001
    epochs = 10

    model = AudioEncoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Initialize variables for training
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # calculating accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)  # Changed from total_loss to total
            correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Initialize variables for testing
        test_loss = 0
        correct = 0
        total = 0
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()

                predicted = torch.argmax(output.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total

        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    # save the model
    torch.save(model.state_dict(), 'audio_encoder_model.pth')

if __name__ == '__main__':
    main()