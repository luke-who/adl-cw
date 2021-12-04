#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ..data import dataset



class AudioCNN(nn.Module):
    def __init__(self, args, in_channels=1):
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
            nn.Dropout(p=args.dropout),
            nn.Linear(3072,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            
            nn.Dropout(p=args.dropout),
            nn.Linear(1000,15)
        )

    def forward(self, x):
        logits = self.cnn_neuro_stack(x)
        return logits
        
class Trainer:

    def __init__(self, model, dataLoaders, args, device, summary_writer):
        self.model = model
        self.train_dataloader, self.test_dataloader = dataLoaders
        self.train_set_size = len(self.train_dataloader.dataset)
        self.train_batches = len(self.train_dataloader)
        self.test_set_size = len(self.test_dataloader.dataset)
        self.test_batches = len(self.test_dataloader)
        self.args = args
        self.device = device
        self.summary_writer = summary_writer
        
    def metrics(self, logits, labels, loss, batch, current_bsize, print_metrics = False):
        batch_size = logits.shape[0]
        catagories = logits.shape[1]
        batch_count = (logits.argmax(dim=1) == labels).sum()
        batch_accuracy = batch_count.item()/batch_size
        class_accuracy = []
        class_count = []
        
        for i in range(catagories):
            preds = logits.argmax(dim=1)
            max_count = (labels == i).sum().item()
            if max_count != 0:
                cat_count = torch.bitwise_and(preds == labels, preds == i).sum().item()
                cat_acc = cat_count/max_count
            else:
                cat_count = 0
                cat_acc = 1
            class_accuracy.append(cat_acc*100)
            class_count.append((cat_count, max_count))
        
        ca_str = ""
        for acc in class_accuracy:
            ca_str = ca_str + f"{acc:3.0f},"
        if print_metrics:
            print(
                f"loss: {loss:>7f}, "
                f"[{batch*self.args.batch_size:>5d}/{self.train_set_size:>5d}], "
                f"batch={batch+1:>3d}/{self.train_batches:>3d}, "
                f"current_bsize={current_bsize:>3d}, "
                f"baccuracy:{batch_accuracy*100:5.1f}%, "
                f"bclass_accuracy:[{ca_str}]"
            )
        return batch_count.item(), class_count
        
    def train(self):        
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch+1}/{self.args.epochs}\n-------------------------------")
            self.model.train()
            
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                logits, loss = self.train_step((X, y))
                
                if batch % self.args.metric_frequency == 0:
                    self.metrics(logits, y, loss, batch, len(X), print_metrics = True)
                    
            self.test()
            # summary_writer.add_scalar("epoch", t, step)
        self.summary_writer.close()
        
    def train_step(self, train_data):
        (X, y) = train_data
        
        # Compute prediction error
        logits = self.model(X)
        loss = self.model.loss_fn(logits, y)

        # Backpropagation
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        return logits, loss.item()
        
    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                test_loss += self.model.loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= self.test_batches
        correct /= self.test_set_size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  
    
def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"dropout={args.dropout}_"
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)
    
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")      

    # Load datasets.
    training_data = dataset.DCASE(Path(args.dataset_root) / "development", 3)
    test_data = dataset.DCASE(Path(args.dataset_root) / "evaluation" , 3)

    # Create data loaders.
    train_dataloader = DataLoader(
        training_data, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    model = AudioCNN(args, in_channels = 10).to(device)
    
    model.loss_fn = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    
    trainer = Trainer(model, (train_dataloader, test_dataloader), args, device, summary_writer)
    trainer.train()
    print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset-root", default="../data/ADL_DCASE_DATA")
    parser.add_argument("--log-dir", default=Path("logs"), type=Path)
    parser.add_argument("--metric-frequency", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    main(parser.parse_args())