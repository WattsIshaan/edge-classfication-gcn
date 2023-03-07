import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset  # not the one from PyG!
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MyDataset(Dataset):
    def __init__(self, path: Path, split_ratio = 0.2, split = 'train', random_seed = 1):
        super().__init__()

        self.data_list = list(path.glob("*/*[0-9].pt"))
        self.data_list = self.data_list[:1000]

        self.split_ratio = split_ratio
        self.split = split
        self.random_seed = random_seed
        
        # Split the data into training and test sets

        train_data_list, rem_data_list = train_test_split(self.data_list, test_size=split_ratio, random_state=random_seed)
        val_data_list, test_data_list = train_test_split(rem_data_list, test_size=0.5, random_state=random_seed)
        
        # Set the data list to the appropriate split
        if split == 'train':
            self.graphs = train_data_list
        elif split == 'test':
            self.graphs = test_data_list
        elif split == 'val':
            self.graphs = val_data_list
        else:
            raise ValueError(f"Invalid split: {split}")
      
    def __getitem__(self, idx):
        return torch.load(self.graphs[idx])
    
    def __len__(self) -> int:
        return len(self.graphs)


class EdgeClassifier(nn.Module):
    def __init__(self, num_node_features=6, num_edge_features=4, hidden_channels=64, num_classes=2):

        super(EdgeClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(num_edge_features + 2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, data):

        # Aggreagation of Node features
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Concatenation of features of nodes and the edge between them
        x = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train(model, device, train_loader, optimizer, criterion):

    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += len(pred)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    print(f'Train Loss : {train_loss}')
    print(f'Train Accuracy: {train_acc}')

    return train_loss, model

def evaluate(model, device, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += len(pred)
    acc = correct / total

    print(f'Validation Accuracy {acc}')

def test(model, device, loader):

    print('Testing The Model')

    model.eval()
    y_true = []
    y_probas = []
    y_pred = []
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            out = model(data)
            y_true += data.y.cpu().numpy().tolist()
            y_pred += out.argmax(dim=1).cpu().numpy().tolist()
            y_probas += out[:, 1].cpu().numpy().tolist()  # probability of class 1

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)
    area = auc(recall, precision)

    print(f'Testing Accuracy {acc}')
    print(f'F1 score: {f1}')
    print(f'AUCPR: {area}')

    return acc, f1, precision, recall, area

def plot_pr_curve(precision, recall, area):
	
    plt.plot(recall, precision)
    plt.plot([1, 0], [0, 1], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve, AUC:{area}')
    plt.save_fig('auc.png')
  

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_class_weights(train_loader):
    sum = 0
    total = 0
    for data in tqdm(train_loader):
        sum += data.y.sum().item()
        total += len(data.y)

    freq1 = sum/total
    freq0 = 1-freq1
    weight1 = 1/(total*freq1)
    weight0 = 1/(total*freq0)

    return torch.tensor([weight0, weight1]).to(get_device())


############################################################################################






















