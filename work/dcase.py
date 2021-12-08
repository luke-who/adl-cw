#!/usr/bin/env python3
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
import os,sys

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

    def __init__(self, model, dataLoaders, categories, args, device, summary_writer):
        self.model = model
        self.train_dataloader, self.test_dataloader = dataLoaders
        self.train_set_size = len(self.train_dataloader.dataset)
        self.train_batches = len(self.train_dataloader)
        self.test_set_size = len(self.test_dataloader.dataset)
        self.test_batches = len(self.test_dataloader)
        self.categories = categories
        self.args = args
        self.device = device
        self.summary_writer = summary_writer
        
    def calc_metrics(self, logits, labels, loss, batch, current_bsize, print_metrics = False, calc_confuMatrix = False):
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
        
    def log_metrics(self, epoch, batch, current_bsize, loss, count, class_count , log_suffix = "train"):
        steps = epoch * self.train_set_size + batch * self.args.batch_size + current_bsize
        self.summary_writer.add_scalar("epoch", epoch, steps)
        self.summary_writer.add_scalars(
                "accuracy",
                {log_suffix: percentageArr(count[0], count[1])},
                steps
        )
        self.summary_writer.add_scalars(
                "class_accuracy_" + log_suffix,
                {str(k): v for k, v in enumerate(percentageArr(class_count[:,0], class_count[:,1]))},
                steps
        )
        self.summary_writer.add_scalars(
                "loss",
                {log_suffix: loss},
                steps
        )
        
    def log_plot(self, epoch, total_confusion_matrix):
        midpoint = (total_confusion_matrix.max() + total_confusion_matrix.min())/2

        fig, ax = plt.subplots(figsize=(8,8))
        ax.matshow(total_confusion_matrix, cmap='Greens')
        ax.set_xticks(range(self.categories))
        ax.set_yticks(range(self.categories))
        # ax.set_xticklabels([])
        ax.set_xlabel("Predicted label.")
        # ax.set_yticklabels([])
        ax.set_ylabel("True label.")
        for (i, j), z in np.ndenumerate(total_confusion_matrix):
            if z > midpoint:
                ax.text(j, i, round(z), color='white', ha='center', va='center')
            else:
                ax.text(j, i, round(z), ha='center', va='center')
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.array(canvas.renderer.buffer_rgba())
        image = image / 255
        image_y, image_x = image.shape[0], image.shape[1]
        image = image[:,:,0:3]
        self.summary_writer.add_image('confusion_matrix', image, epoch, dataformats='HWC')
        
    def train(self):        
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch+1}/{self.args.epochs}\n-------------------------------")
            self.model.train()
            
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                logits, loss = self.train_step((X, y))
                
                if batch % self.args.metric_frequency == 0:
                    batch_count, class_count, _ = self.calc_metrics(logits, y, loss, batch, len(X), print_metrics = True)
                    self.log_metrics(epoch, batch, len(X), loss, batch_count, class_count)
                    
            total_batch_count, total_class_count, total_loss, total_confusion_matrix = self.test()
            self.log_metrics(epoch, batch, len(X), total_loss, total_batch_count, total_class_count, log_suffix = "test")
            self.log_plot(epoch, total_confusion_matrix)
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
        total_loss = 0
        total_batch_count = np.zeros((2))
        total_class_count = np.zeros((self.categories, 2))
        total_confusion_matrix = np.zeros((self.categories, self.categories))
            
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.test_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.model.loss_fn(logits, y).item()
                batch_count, class_count, confusion_matrix = self.calc_metrics(logits, y, loss, batch, len(X), calc_confuMatrix = True)
                
                total_batch_count += batch_count
                total_class_count += class_count
                total_loss += loss
                total_confusion_matrix += confusion_matrix
                
        total_loss /= self.test_batches
        total_acc =  percentageArr(total_batch_count[0], total_batch_count[1])
        total_class_acc = percentageArr(total_class_count[:,0], total_class_count[:,1])
        total_class_acc = formatCAList(total_class_acc)
        print(
            f"Test Error: \n Accuracy: {(total_acc):>0.1f}%, "
            f"Avg loss: {loss:>8f}, "
            f"class_accuracy:[{total_class_acc}]\n"
        )
        return total_batch_count, total_class_count, total_loss, total_confusion_matrix
    
def get_summary_writer_log_dir(args: argparse.Namespace, command_prefix = "") -> str:
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"{args.prefix[1:-1]}"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_" +
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
    training_data = dataset.DCASE_clip(Path(args.dataset_root) / "development", 3, normData = True)
    test_data = dataset.DCASE_clip(Path(args.dataset_root) / "evaluation" , 3, normData = True, priorNorm = training_data.prior_norm())

    # Calculate total number of classes/categories
    # train_data = dataset.DCASE(Path(args.dataset_root) / "development",3)
    # sample_classes = [train_data[i][1] for i in range(len(train_data))]
    # categories = len(np.unique(sample_classes))
    categories = 15
    # print("Total number of classes/categories:",categories)

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
    model = AudioCNN(args, categories = categories, in_channels = 1).to(device)
    
    model.loss_fn = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    
    trainer = Trainer(model, (train_dataloader, test_dataloader), categories, args, device, summary_writer)
    trainer.train()
    print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", type=ascii)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset-root", default="../data/ADL_DCASE_DATA")
    parser.add_argument("--log-dir", default=Path("logs"), type=Path)
    parser.add_argument("--metric-frequency", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=5e-4, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    main(parser.parse_args())