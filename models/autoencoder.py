import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self):
        super(TimeSeriesAutoencoder, self).__init__()

        # Encoder (4 layers fully connected)
        self.encoder = nn.Sequential(
            nn.Linear(204, 128),
            nn.Tanh(),

            nn.Linear(128, 64),
            nn.Tanh(),

            nn.Linear(64, 32),
            nn.Tanh(),
        )

        # Decoder (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),

            nn.Linear(64, 128),
            nn.Tanh(),

            nn.Linear(128, 204)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded