#!/usr/bin/env python3
from torch import nn

class AudioCNN(nn.Module):
    def __init__(self, args, categories, in_channels=1):
        super().__init__()
        self.cnn_neuro_stack = nn.Sequential(
            
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=(5, 5),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(5, 5),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            
            
            nn.Flatten(),
            # nn.Dropout(p=args.dropout),
            # nn.Linear(3072,1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            
            nn.Dropout(p=args.dropout),
            nn.Linear(3072, categories)
        )

    def forward(self, x):
        logits = self.cnn_neuro_stack(x)
        return logits