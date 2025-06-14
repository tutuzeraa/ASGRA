import os
import json 
import logging

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import GATv2Conv

from datasets.places8 import Places8SceneGraphDataset


logging.basicConfig(
    level=logging.INFO,                           
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)             

data_dir = "./data/graphs"

def main():
    # edge_index = torch.tensor([[0,1,1,2], [1,0,2,1]], dtype=torch.long)
    # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    # data = Data(x=x, edge_index=edge_index)
    # print(data)

    dataset = Places8SceneGraphDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size = 1, shuffle=True)

    print(next(iter(train_loader)))


if __name__ == "__main__":
    main()

