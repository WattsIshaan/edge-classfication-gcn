# Edge classification problem using GNN

## Problem Statement

Task is to train an Edge Classifier Graph Neural Network to classify the edges (given by the edge_index) as true or false (given by the array y). The inputs to training/inference are x (the node features) and edge_features.

## Dataset

The dataset used can be found on this link[href=https://cernbox.cern.ch/s/YQxujEYrVFFpylN]. The dataset is split up in 10 gzip compressed tarballs. Download and extract the dataset and run the .ipynb files in the directory containing the data folders.

## requirements.txt

Install the required dependencies. The model is built using pytorch-geometric which needs to be installed. Specify your torch and cuda version in the speicfied brackets.

```pip
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html 
pip install torch-geometric
```

For example with ```torch==1.13.1``` and ```cuda==11.6```

```pip
!pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
!pip install torch-geometric
```

## Model Architecture

1. GCNConv layers are used for message passing between nodes in the graph. We use **two GCNConv layers** to extract node features.

2. After message passing, we **concatenate the node features with the edge features** to get a combined feature vector for each edge.

3. We then apply **two fully connected layers** to classify the edges, with a ReLU activation function and dropout regularization.

## Description of the files

This repository contains two ```.ipynb``` notebooks - ```GNN-Tracking and GNN-Tracking-Module```. It also contains a ```cern.py``` file.

1. GNN-Tracking.ipynb <br>
It contains the code with step by step description of the whole project.

2. GNN-Tracking-Module.ipynb and cern.py <br>
```cern.py``` contains all the functions and classes used in the project which makes the code reusable. ```GNN-Tracking-Module.ipynb``` is the same implementation of the project built by importing the classes and methods from the ```.py file```.

