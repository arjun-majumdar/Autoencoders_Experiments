

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle


"""
Reading CSV file in PyTorch.


References-
https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1
https://www.kaggle.com/code/pinocookie/pytorch-dataset-and-dataloader/notebook
"""


print(f"torch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Number of GPU(s) available = {torch.cuda.device_count()}")
else:
    print("PyTorch does not have access to GPU")

# Device configuration-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device is {device}')
'''
torch version: 1.11.0
PyTorch does not have access to GPU
Available device is cpu
'''


# Read in data CSV using pandas-
# data = pd.read_csv("Downloads/yahoo_stock.csv")

# data.shape
# (1825, 7)

# data.dtypes
'''
Date          object
High         float64
Low          float64
Open         float64
Close        float64
Volume       float64
Adj Close    float64
dtype: object
'''

# This does not work due to 'Date' attribute which is string-
# data_np = np.loadtxt("Downloads/yahoo_stock.csv", dtype = np.float32, delimiter = ',', skiprows = 1)

# To exclude 'Date' feature, use-
# data.iloc[:, 1:].values

# Convert from numpy to torch tensor-
# data_torch = torch.from_numpy(data.iloc[:, 1:].values, dtype = torch.float32)

# Sanity checks-
# data_torch.shape
# torch.Size([1825, 6])

# data_torch[:5, :]

# data_torch.dtype
# torch.float64


class TimeSeriesDataset(Dataset):
    def __init__(self, filename):
        data = pd.read_csv(filename)
        
        # Convert 'Dae' column to datetime-
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort in increasing order for 'Date' column-
        data.sort_values(by = 'Date', ascending = True, inplace = True)
        
        # Reset indices-
        data.reset_index(drop = True, inplace = True)
        
        # Write any data pre-processing steps here.

        # Initialize a MinMax scaler-
        mm_scaler = MinMaxScaler(feature_range = (0, 1))
        data_scaled = mm_scaler.fit_transform(data.iloc[:, 1:].to_numpy())
        
        # Convert from np array to pd.DataFrame-
        # data_scaled = pd.DataFrame(data_scaled, columns = cols)
        
        # Convert to torch tensors-
        self.data_scaled = torch.tensor(data_scaled, dtype = torch.float32)
        
        # Save MinMaxScaler for later use/inverse scaling-
        with open("Trained_MinMaxScaler.pickle", "wb") as file:
            pickle.dump(mm_scaler, file)
    
    
    def __len__(self):
        return len(self.data_scaled)
    
    
    def __getitem__(self, idx):
        return self.data_scaled[idx]


# Initialize time-series dataset-
data_torch = TimeSeriesDataset(filename = "Downloads/yahoo_stock.csv")

# Sanity checks-
len(data_torch)
# 1825

# data_torch[:5, :]


# Create data loader
data_loader = DataLoader(dataset = data_torch, batch_size = 256, shuffle = False)
# shuffle = False due to time-series data!

print(f"len(data_loader) = {len(data_loader)}")
# len(data_loader) = 8

len(data_torch) / 256
# 7.12890625

# Get a batch of data-
x = next(iter(data_loader))

print(f"batch-size shape: {x.shape}")
# batch-size shape: torch.Size([256, 6])


"""
# Another solution-

class MyDataset(Dataset):
    def __init__(self, root, n_inp):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        self.x , self.y = (
            torch.from_numpy(self.data[:,:n_inp]),
            torch.from_numpy(self.data[:,n_inp:])
        )
                           
    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx,:]
        
    def __len__(self):
        return len(self.data)


myData = MyDataset("data.csv", 20)

data_loader = DataLoader(myData, batch_size = 4, shuffle = True)


for x,y in data_loader:
   # iterate/do something


# Note: You should do any pre-processing/transforms in the '__getitem__()' method
# if your dataset (images/signals) is huge.

You need to subclass dataset class and the create your own dataset class. Use that
particular dataset object to create a dataloader. check the official pytorch docs.
They have a very good example.
"""

