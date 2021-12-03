#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

class DCASE(Dataset):
    def __init__(self, root_dir: str, clip_duration: int):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        # print(self._labels.label.astype('category'))
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s

        self._data_len = len(self._labels)

    def __getitem__(self, index):
        #reading spectrograms
        filename, label = self._labels.iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec
        spec = self.__trim__(spec)
        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            #spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len

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
    
def train(dataloader, model, loss_fn, optimizer, args, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % args.print_frequency == 0:
            loss, current = loss.item(), batch * args.batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], batch={batch:>3d}/{num_batches:>3d}, current bsize={len(X):>3d}")
    return num_batches
    
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
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
    training_data = DCASE(Path(args.dataset_root) / "development", 3)
    test_data = DCASE(Path(args.dataset_root) / "evaluation" , 3)

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
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    
    step = 0
    for t in range(args.epochs):
        print(f"Epoch {t+1}/{args.epochs}\n-------------------------------")
        steps = train(train_dataloader, model, loss_fn, optimizer, args, device)
        test(test_dataloader, model, loss_fn, device)
        summary_writer.add_scalar("epoch", t, step)
        step += steps
    summary_writer.close()
    print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset-root", default="../ADL_DCASE_DATA")
    parser.add_argument("--log-dir", default=Path("logs"), type=Path)
    parser.add_argument("--print-frequency", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    main(parser.parse_args())