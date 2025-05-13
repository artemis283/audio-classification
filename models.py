import torch.nn as nn
import torch
import numpy as np
import librosa

# maybe add weight decay L2 regularization
# replace maxpool with average pooling

class AudioConvModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioConvModel, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(43008, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)  # Add dropout
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Add dropout
        
        x = self.fc2(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=1501):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # this is the classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) # classification token

        positional_encoding = torch.zeros(max_seq_len, embedding_dim)

        for pos in range(max_seq_len):
            for i in range(embedding_dim):
                if i % 2 == 0:
                    positional_encoding[pos][i] = np.sin(pos / (10000 ** (i / embedding_dim))) # if i is even use sin
                else:
                    positional_encoding[pos][i] = np.cos(pos / (10000 ** ((i - 1) / embedding_dim))) # if i is odd use cos

        self.register_buffer("positional_encoding", positional_encoding) # part of the model but not a parameter to be updated ie it is non-trainable 

    def forward(self, x):
        batch_size = x.size(0)

        # expand class token to have a class token for every image in the batch
        tokens_batch = self.cls_token.expand(batch_size, -1, -1)

        # add tokens to the beginning of each embedding
        x = torch.cat((tokens_batch, x), dim=1)

        positional_encoding = self.positional_encoding[:x.size(1), :]

        x = x + self.positional_encoding # add the positional encoding to the patch embeddings

        return x
    

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=80, n_time=3000, n_channels=1, batch_size=32, dim=384, r_mlp=2, num_classes=10):
        super(AudioEncoder, self).__init__()

        self.gelu = nn.GELU()

        # number of mels is in channles as this contains the number of features at each time step
        self.conv1 = nn.Conv1d(in_channels=n_mels, out_channels=dim,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)

        # positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim=dim)

        # residual attention/transformer encoder blocks
        self.attention = nn.MultiheadAttention(dim, num_heads=8)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # classification head and MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, r_mlp * dim),
            nn.GELU(),
            nn.Linear(r_mlp * dim, dim),
        )

        self.classifier = nn.Sequential(nn.Linear(dim, num_classes),
                                        nn.Softmax(dim=1))

    
    def forward(self, x):
        # shape = (batch_size, 80, 3000)
        x = self.gelu(self.conv1(x))

        # shape = (batch_size, 384, 1500)
        x = self.gelu(self.conv2(x))

        # shape = (batch_size, 384, 1500)
        x = x.transpose(1, 2) 

        # shape = (batch_size, 1500, 384)
        x = self.positional_encoding(x)

        # shape = (1500, batch_size, 384)
        x = x.transpose(0, 1)

        # shape = (batch_size, 1500, 384)
        attention_output, _ = self.attention(x, x, x) # ignoring the weights in the tuple ouputted by the attention layer
        out = self.ln1(x + attention_output)

        # shape = (batch_size, 1500, 384)
        out = self.ln2(out + self.mlp(out))
        
        # shape = (batch_size, 384)

        out = self.classifier(out[0])
        # shape = (batch_size, 10)

        return out
        
if __name__ == "__main__":
    model = AudioEncoder()
    print(model)

    dim = 384
    batch_size = 1

    x = torch.randn(batch_size, 80, 3000)
    out = model(x)

    print(out.shape)


        