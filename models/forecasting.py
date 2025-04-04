import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

class TimeSeriesForecaster(nn.Module):
    def __init__(self):
        super(TimeSeriesForecaster, self).__init__()

        self.window_size = config["forecasting"]["window_size"]
        self.num_features = config["forecasting"]["num_features"]
        self.global_features = config["forecasting"]["global_features"]

        # Branch 1 (Time Series (LSTM))
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.lstm_dropout = nn.Dropout(0.1)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(64, self.num_features),
            nn.Softmax(dim=1)
        )

        # Branch 2 (Global Features)
        self.global_net = nn.Sequential(
            nn.Linear(self.global_features, 8),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
            nn.Tanh()
        )

        # Fusion & Output
        self.fusion = nn.Linear(64 + 4, 10)  # 64 (LSTM) + 4 (global) → 10 features

    def forward(self, x_time_series, x_global):
        lstm_out, _ = self.lstm(x_time_series)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.lstm_dropout(lstm_out)
        if config["forecasting"]["use_attention"]:
            attn_weights = self.attention(lstm_out)
            lstm_out = lstm_out * attn_weights.unsqueeze(1)

        global_out = self.global_net(x_global)

        combined = torch.cat((lstm_out, global_out), dim=1)
        output = self.fusion(combined)

        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesForecaster().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.HuberLoss(delta=1.0)

print(model)