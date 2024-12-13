import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """RNN model for MNIST classification treating images as sequences of rows."""

    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2

    def __init__(self):
        super().__init__()

        # Input size will be 28 features (width) for each of the 28 rows
        self.input_size = 28
        self.hidden_size = self.HIDDEN_SIZE

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            dropout=self.DROPOUT_RATE if self.NUM_LAYERS > 1 else 0
        )

        # Batch normalization for LSTM outputs
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.DROPOUT_RATE)

        # Final classification layer
        self.fc = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        # Need to reshape to: (batch_size, sequence_length, input_size)
        # Where sequence_length = height (28) and input_size = width (28)

        # Remove the channel dimension and reshape
        x = x.squeeze(1)  # Remove channels dimension
        # x is now (batch_size, height, width)

        # Pass through LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # We only need the last output for classification
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Apply batch normalization
        normalized = self.batch_norm(last_output)

        # Apply dropout
        dropped = self.dropout(normalized)

        # Final classification
        logits = self.fc(dropped)

        return F.log_softmax(logits, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)