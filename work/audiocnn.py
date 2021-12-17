#!/usr/bin/env python3
from torch import nn
import argparse

class AudioCNN(nn.Module):
    def __init__(
        self, 
        args: argparse.Namespace, 
        categories: int, 
        in_channels: int=1
    ):  
        self.loss_fn = None
        self.optimizer = None
        super().__init__()
        self.cnn_neuro_stack = nn.Sequential(  

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=(5, 5),
                padding=(2, 2),
                # padding preserves the image size after convolution
            ),
            nn.BatchNorm2d(128),
            # batch normalisation make sure inputs & weights don't become exetremely imbalanced
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            # nn.AdaptiveMaxPool2d(),
            # We use pooling to reduce each feature map dimensions and to enhance the network invariance to input pattern shifts
            # Max pooling reduces the resolution of the output of a given convolutional layer. Therefore it 
            # reduces the number of parameters in the network, this in turn reduces computational load, 
            # it may also help to reduce overfitting as it'll extract&preserve the most important features
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(5, 5),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            # nn.AdaptiveMaxPool2d(),
            
            
            # nn.Flatten(),
            nn.Dropout(p=args.dropout),
            nn.Linear(3072,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            
            nn.Dropout(p=args.dropout),
            nn.Linear(3072, categories)
        )

    def forward(self, x):
        logits = self.cnn_neuro_stack(x)
        return logits
