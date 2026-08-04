import torch.nn as nn


# ============================================================================
# MobilenetLSTM model definition
# ============================================================================

class MobilenetLSTM(nn.Module):
    """MobilenetLSTM model for violence classification."""

    def __init__(self, num_classes, seq_length, lstm_dropout=0):
        super().__init__()
        from torchvision.models import mobilenet_v2
        mobilenet = mobilenet_v2(weights=None)
        mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
        self.feature_extractor = mobilenet
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=32, num_layers=1,
            batch_first=True, bidirectional=True, dropout=lstm_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.feature_extractor(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

