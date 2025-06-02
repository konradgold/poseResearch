import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)  # Expecting shape (N, 17, 3)
        assert len(self.data.size()) == 3
        assert self.data.size(1) == 17
        assert self.data.size(2) == 3
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class PoseRegressor(torch.nn.Module):  # Renamed to Regressor
    def __init__(self):
        super(PoseRegressor, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(17 * 3, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),  # Changed to output single value
            torch.nn.Sigmoid()  # Added to constrain output between 0 and 1
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x).squeeze()

def create_dataloader(data, labels, batch_size=32, shuffle=True):
    dataset = PoseDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
