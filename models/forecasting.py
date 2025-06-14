import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TimeSeriesForecaster(nn.Module):
    def __init__(self):
        super(TimeSeriesForecaster, self).__init__()

        self.num_features = 10
        self.global_features = 4

        # Branch 1 (Time Series (LSTM))
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.lstm_dropout = nn.Dropout(0.1)

        self.lstm_to_features = nn.Linear(64, self.num_features)

        # Attention
        self.attention_weights_generator = nn.Sequential(
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
        self.fusion = nn.Linear(10 + 4, 10)  # 64 (LSTM) + 4 (global) → 10 features

    def forward(self, x_time_series, x_global):
        lstm_out, _ = self.lstm(x_time_series)
        lstm_out_last_step = lstm_out[:, -1, :]
        lstm_out_last_step = self.lstm_dropout(lstm_out_last_step)

        lstm_features_projected = self.lstm_to_features(lstm_out_last_step)
        attn_weights = self.attention_weights_generator(lstm_out_last_step)
        lstm_final_output = lstm_features_projected * attn_weights

        global_out = self.global_net(x_global)

        combined = torch.cat((lstm_final_output, global_out), dim=1)
        output = self.fusion(combined)

        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesForecaster().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.HuberLoss(delta=1.0)

print(model)