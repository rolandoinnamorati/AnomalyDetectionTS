import torch
import torch.nn as nn
import torch.optim as optim

input_length = 20   # Rolling window length
num_features = 14   # 4 temporal features + 10 for each data
input_dim = input_length * num_features  # Total size of inputs

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self):
        super(TimeSeriesAutoencoder, self).__init__()

        # Encoder (4 layers fully connected)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),  # Bottleneck
            nn.Tanh()
        )

        # Decoder (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesAutoencoder().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(model)