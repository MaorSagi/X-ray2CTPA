import torch
import torch.nn as nn
import numpy as np
class ChestXRayModel(nn.Module):
    def __init__(self, embeddings_size=1376, hidden_layer_sizes=[512,256], dropout=0.2):
        super(ChestXRayModel, self).__init__()

        self.embeddings_size = embeddings_size
        # Reshaping layer (flatten spatial dimensions for pooling)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, embeddings_size))
        layers = []
        input_size = embeddings_size
        for size in hidden_layer_sizes:
            layers.extend([
                nn.Linear(input_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            input_size = size
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.squeeze(
            1)  # From [batch_size, 1, token_num, spatial_dim, embedding_size] to ([)batch_size, token_num, spatial_dim, embedding_size)
        # Reshape input embeddings (batch_size, token_num, spatial_dim, embedding_size)
        x = self.global_pooling(x).squeeze(
            2)  # Global pooling across spatial dimension (batch_size, token_num, embedding_size)
        x = x.mean(dim=1)  # Average pooling across tokens (batch_size, embedding_size)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)

        return x