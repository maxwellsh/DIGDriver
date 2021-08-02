import torch
import torch.nn as nn

class Autoencoder_FC(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        bs,w,tracks = in_shape
        self.encoder = nn.Sequential(
            nn.Linear(w * tracks, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, w * tracks),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embeding(self, x):
        x = self.encoder(x)
        return x

class Mean_Vec_Autoencoder_FC(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        bs,w,tracks = in_shape
        self.encoder = nn.Sequential(
            nn.Linear(tracks, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, tracks),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embeding(self, x):
        x = self.encoder(x)
        return x
