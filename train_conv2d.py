import soundata
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
from models import AudioConvModel
import torch.nn as nn
dataset = soundata.initialize('urbansound8k')

# loading the .csv file
metadata = pd.read_csv('/Users/artemiswebster/sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')


class UrbanSoundDataset(Dataset):
    def __init__(self, clip_duration=4, sr=22050, n_mels=128, fmax=8000, metadata=metadata):
        self.dataset = metadata
        self.clip_duration = clip_duration
        self.sr = sr
        self.n_mels = n_mels
        self.fmax = fmax
        self.metadata = metadata
        self.audio_dir = '/Users/artemiswebster/sound_datasets/urbansound8k/audio/'

        self.classes = sorted(metadata['class'].unique())
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        filename = self.metadata.iloc[idx]['slice_file_name']
        label = self.metadata.iloc[idx]['class']
        fold = self.metadata.iloc[idx]['fold']

        audio_path = os.path.join(self.audio_dir, f'fold{fold}', filename)

        clip, sr = librosa.load(audio_path, sr=self.sr)

        target_len = self.clip_duration * self.sr
        if len(clip) < target_len:
            clip = np.pad(clip, (0, target_len - len(clip)))
        else: 
            clip = clip[:target_len]

        mel_spectrogram = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=self.n_mels, fmax=self.fmax)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        log_mel_spectrogram = torch.FloatTensor(log_mel_spectrogram).unsqueeze(0)
        
        label = self.class_to_index[label]
        label = torch.tensor(label, dtype=torch.long)

        return log_mel_spectrogram, label


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

    model = AudioConvModel()
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
    torch.save(model.state_dict(), 'audio_conv_model.pth')

if __name__ == '__main__':
    main()

