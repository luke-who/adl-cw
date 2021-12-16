#!/usr/bin/env python3
from typing import Tuple,Optional
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from pathlib import Path
import os,sys,copy

#******* Setting path for importing data from parent directory*******#
# Getting the name of the directory where the this file is present.
currentdir = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parentdir = os.path.dirname(currentdir)
# Adding the parent directory to the sys.path.
sys.path.append(parentdir)
# Now we can import the module from the parent directory.
from data import dataset
from audiocnn import AudioCNN
        
class Trainer:
    r"""Training class which contains train and test methods.

    Args: 
    """
    def __init__(
        self, 
        model: nn.Module, 
        dataLoaders: Tuple[DataLoader,DataLoader], 
        categories: int,
        categories_type, 
        args: argparse.Namespace, 
        device: torch.device, 
        summary_writer: SummaryWriter
    ):
        self.model = model.to(device)
        self.train_dataloader, self.train_split_dataloader, self.valid_split_dataloader, self.test_dataloader = dataLoaders
        self.train_set_size = len(self.train_dataloader.dataset) # total number of datapoints in train set/development folder
        self.train_batches = len(self.train_dataloader) # number of batches in train set
        self.test_set_size = len(self.test_dataloader.dataset) # total number of datapoints in test set/evaluation folder
        self.test_batches = len(self.test_dataloader) # number of batches in test set
        self.categories = categories # number of classes/categories
        self.categories_type = categories_type
        self.args = args
        self.device = device
        self.summary_writer = summary_writer
        
    def calc_metrics(
        self, 
        logits: torch.Tensor, # prediction/output
        labels: torch.Tensor, 
        loss: float, 
        batch: int, 
        current_bsize: int, 
        print_metrics: Optional[bool] = False, 
        calc_confuMatrix: Optional[bool] = False
    ):
        batch_size = logits.shape[0]
        categories = logits.shape[1]
        batch_count = (logits.argmax(dim=1) == labels).sum()
        batch_accuracy = batch_count.item()/batch_size
        class_accuracy = []
        class_count = []
        confusion_matrix = np.zeros((categories, categories))
        
        for i in range(categories):
            preds = logits.argmax(dim=1)
            matched_labels = labels == i
            if calc_confuMatrix:
                for j in range(categories):
                    matched_preds = preds == j
                    confusion_matrix[j,i] = (matched_preds & matched_labels).sum().item()
            
            max_count = matched_labels.sum().item()
            if max_count != 0:
                cat_count = ((preds == labels) & (preds == i)).sum().item()
                cat_acc = cat_count/max_count
            else:
                cat_count = 0
                cat_acc = 1
            class_accuracy.append(cat_acc*100)
            class_count.append((cat_count, max_count))
        
        ca_str = formatCAList(class_accuracy)
        if print_metrics:
            print(
                f"loss: {loss:>7f}, "
                f"[{batch*self.args.batch_size:>5d}/{self.train_set_size:>5d}], "
                f"batch={batch+1:>3d}/{self.train_batches:>3d}, "
                f"current_bsize={current_bsize:>3d}, "
                f"baccuracy:{batch_accuracy*100:5.1f}%, "
                f"bclass_accuracy:[{ca_str}]"
            )
        return np.array([batch_count.item(), batch_size]), np.array(class_count), confusion_matrix
        
    def log_metrics(
        self, 
        epoch: int,
        step: int, 
        loss: float, 
        count: int, 
        class_count: int, 
        log_suffix: str = "train"
    ):
        self.summary_writer.add_scalar("epoch", epoch, step)
        self.summary_writer.add_scalars(
                "accuracy",
                {log_suffix: percentageArr(count[0], count[1])},
                step
        )
        self.summary_writer.add_scalars(
                "class_accuracy_" + log_suffix,
                {str(k): v for k, v in enumerate(percentageArr(class_count[:,0], class_count[:,1]))},
                step
        )
        self.summary_writer.add_scalars(
                "loss",
                {log_suffix: loss},
                step
        )
        
    def log_plot(self, epoch, total_confusion_matrix, log_suffix = "test"):
        midpoint = (total_confusion_matrix.max() + total_confusion_matrix.min())/2

        fig, ax = plt.subplots(figsize=(8,8))
        ax.matshow(total_confusion_matrix, cmap='Greens')
        ax.set_xticks(range(self.categories))
        ax.set_yticks(range(self.categories))
        ax.set_xticklabels(self.categories_type)
        ax.set_xlabel("Predicted label.")
        ax.set_yticklabels(self.categories_type)
        ax.set_ylabel("True label.")
        for (i, j), z in np.ndenumerate(total_confusion_matrix):
            if z > midpoint:
                ax.text(j, i, "{:.0f}".format(z), color='white', ha='center', va='center')
            else:
                ax.text(j, i, "{:.0f}".format(z), ha='center', va='center')
        plt.xticks(rotation=90, ha='center')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.array(canvas.renderer.buffer_rgba())
        image = image / 255
        image_y, image_x = image.shape[0], image.shape[1]
        image = image[:,:,0:3]
        self.summary_writer.add_image('confusion_matrix_' + log_suffix, image, epoch, dataformats='HWC')
        
    def training_loop(self):
        self.nonfull_training(valid_freq = self.args.valid_frequency, max_worsen_streak = self.args.max_worsen_streak)
        self.full_training(self.args.epochs)
        self.summary_writer.close()
        
    def nonfull_training(self, valid_freq = 2, max_worsen_streak = 5, epoch_limit=200):
        print("Non-full training.")
        self.test(self.valid_split_dataloader, 0, 0, log_suffix = "nonfull_test")
        best_valid_acc = self.test(self.valid_split_dataloader, 0, 0, log_suffix = "nonfull_validation")
        print("first layer weight sum:", self.model.state_dict()['cnn_neuro_stack.0.weight'].sum().item())
        best_model = copy.deepcopy(self.model)
        worsen_streak = 0
        for epoch in range(0, epoch_limit):
            print(f"Epoch {epoch+1}/{epoch_limit}")
            print("-------------------------------------------------------------")
            self.model.train()
            
            for batch, (X, y) in enumerate(self.train_split_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                current_bsize = len(X)
                step = epoch * len(self.train_split_dataloader) + batch
                logits, loss = self.train_step((X, y)) # X is the feature, y is the the true label

                if batch % self.args.metric_frequency == 0:
                    count, class_count, _ = self.calc_metrics(logits, y, loss, batch, current_bsize, print_metrics = True)
                    self.log_metrics(epoch, step, loss, count, class_count, log_suffix = "nonfull_train")
            
            if epoch % valid_freq == 0:
                current_valid_acc = self.test(self.valid_split_dataloader, epoch, step, log_suffix = "nonfull_validation")      
                if current_valid_acc > best_valid_acc:
                    print("Current model better, updating.")
                    best_model = copy.deepcopy(self.model.cpu())
                    best_valid_acc = current_valid_acc
                    worsen_streak = 0
                else:
                    print(f"Current model worse, regressing (worsen_streak={worsen_streak}).")
                    self.model = best_model.to(device)
                    worsen_streak += 1
                    if worsen_streak >= max_worsen_streak:
                        print(f"worsen_streak = {worsen_streak}, terminating non-full training.")
                        break
            print("first layer weight sum:", self.model.state_dict()['cnn_neuro_stack.0.weight'].sum().item())
            self.test(self.test_dataloader, epoch, step, log_suffix = "nonfull_test")
        
    def full_training(self, epochs):
        print("Full training for " + str(epochs) + " epochs.")
        self.test(self.test_dataloader, 0, 0, log_suffix = "full_test")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print("-------------------------------------------------------------")
            self.model.train()
            
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                current_bsize = len(X)
                step = epoch * len(self.train_dataloader) + batch
                logits, loss = self.train_step((X, y)) # X is the feature, y is the the true label

                if batch % self.args.metric_frequency == 0:
                    count, class_count, _ = self.calc_metrics(logits, y, loss, batch, current_bsize, print_metrics = True)
                    self.log_metrics(epoch, step, loss, count, class_count, log_suffix = "full_train")
            self.test(self.test_dataloader, epoch, step, log_suffix = "full_test")
        
    def train_step(self, train_data: Dataset):
        r"""Train data with forward and backward propagation.

        Args:
            train_data (Dataset): train dataset.
        """
        (X, y) = train_data
        
        # Compute prediction error
        logits = self.model(X)
        loss = self.model.loss_fn(logits, y)

        # Backpropagation
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        return logits, loss.item()
        
    def test(self, test_dataloader, epoch=0, step=0, log_suffix = "test"):
        print("Evaluating model for " + log_suffix + " metrics.")
        self.model.eval()
        batches = len(test_dataloader)
        clips = test_dataloader.dataset._num_clips
        batch_logits = torch.zeros([batches, self.categories])
        batch_labels = torch.zeros([batches], dtype = torch.long)
        loss = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss += self.model.loss_fn(logits, y).item()
                batch_logits[batch] = logits.mean(dim=0)
                batch_labels[batch] = y[0]
            loss /= batches
            total_count, total_class_count, total_confusion_matrix = self.calc_metrics(batch_logits, batch_labels, loss, 0, batches, calc_confuMatrix = True)
            
        total_acc =  percentageArr(total_count[0], total_count[1])
        total_class_acc = percentageArr(total_class_count[:,0], total_class_count[:,1])
        total_class_acc = formatCAList(total_class_acc)
        print(
            f"{log_suffix} Error: \n Accuracy: {(total_acc):>0.1f}%, "
            f"Avg loss: {loss:>8f}, "
            f"class_accuracy:[{total_class_acc}]\n"
        )
        self.log_metrics(epoch, step, loss, total_count, total_class_count, log_suffix = log_suffix)
        self.log_plot(epoch, total_confusion_matrix, log_suffix = log_suffix)
        return total_acc
    
def get_summary_writer_log_dir(args: argparse.Namespace, command_prefix = "") -> str:
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"{args.prefix[1:-1]}"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"train_ratio={args.train_ratio}"
        f"valid_frequency={args.valid_frequency}_"
        f"max_worsen_streak={args.max_worsen_streak}_" +
        (f"dropout={args.dropout}_" if args.dropout!=0 else "") +
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def formatCAList(class_accuracy):
        ca_str = ""
        for acc in class_accuracy:
            ca_str = ca_str + f"{acc:3.0f},"
        return ca_str
        
def percentageArr(count, max):
    if max != 0:
        return count*100/max
    else:
        return 100
percentageArr = np.vectorize(percentageArr)
    
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")      

    # Load datasets.
    training_data = dataset.DCASE_clip(Path(args.dataset_root) / "development", 3, offSet = True, normData = True)
    train_split, valid_split = training_data.split(train_rat = args.train_ratio)
    test_data = dataset.DCASE_clip(Path(args.dataset_root) / "evaluation" , 3, normData = True, priorNorm = training_data.prior_norm())

    # Calculate total number of classes/categories
    categories = len(training_data.categories)
    print("Total number of classes/categories:",categories)

    # Create data loaders.
    # Training dataloaders
    train_dataloader = DataLoader(
        training_data, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    train_split_dataloader = DataLoader(
        train_split, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    # Verification dataloaders
    valid_split_dataloader = DataLoader(
        valid_split, 
        batch_size = valid_split._num_clips,
        shuffle=False,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size = test_data._num_clips,
        shuffle=False,
        pin_memory=True
    )
    model = AudioCNN(args, categories = categories, in_channels = 1)
    
    model.loss_fn = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    dataLoaders = train_dataloader, train_split_dataloader, valid_split_dataloader, test_dataloader
    trainer = Trainer(model, dataLoaders, categories, training_data.categories, args, device, summary_writer)
    trainer.training_loop()
    print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", type=ascii)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset-root", default="../data/ADL_DCASE_DATA")
    parser.add_argument("--log-dir", default=Path("logs"), type=Path)
    parser.add_argument("--metric-frequency", default=1, type=int)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--valid-frequency", default=2, type=int)
    parser.add_argument("--max-worsen-streak", default=5, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=5e-4, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    main(parser.parse_args())