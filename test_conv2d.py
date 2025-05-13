# testing the convultional model 
from torch.utils.data import DataLoader
import pandas as pd
import torch
from models import AudioConvModel
from train_conv2d import UrbanSoundDataset, get_fold_data

metadata = pd.read_csv('/Users/artemiswebster/sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')

audio_conv_model = AudioConvModel()
audio_conv_model.load_state_dict(torch.load('audio_conv_model.pth'))

audio_conv_model.eval()

test_dataset = get_fold_data(metadata, folds=[10])
test_dataset = UrbanSoundDataset(metadata=test_dataset)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_conv_model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = audio_conv_model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for pred, label in zip(preds, labels):
                print(f'Pred: {pred}, Label: {label}')

if __name__ == '__main__':
    main()
            

            
            


