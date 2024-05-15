"""
Straight Through Estimator example using Conv-Autoencoder + MNIST + PyTorch

Suppose that we want to binarize the activations of a layer using the function-
f(x) = 1, if x > 0, else 0

The problem with this function is that its gradients are 0s. To avoid this
problem, we use a STE in the backward propagation.

A STE estimates the gradients of a function. Specifically, it ignores the
derivative of the threshold function and passes on the incoming gradients
as if the function was an identity function.

A STE makes the gradient of the threshold function look like the gradient
of the identity function.


Refer-
https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pickle, functools


print(f"torch version: {torch.__version__}")

# Check if there are multiple devices (i.e., GPU cards)-
print(f"Number of GPU(s) available = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch does not have access to GPU")

# Device configuration-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device is {device}')


# Specify hyper-parameters-
num_epochs = 100
batch_size = 1024
learning_rate = 0.001


# MNIST Data Pre-Processing-
# Define transformations for MNIST dataset-
# MNIST dataset statistics-
mean = np.array([0.1307])
std_dev = np.array([0.3081])

transforms_apply = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std_dev)
    ]
)

path_to_data = "/home/majumdar/Downloads/.data/"

# Load MNIST dataset-
train_dataset = torchvision.datasets.MNIST(
    # root = './data', train = True,
    root = path_to_data + "data", train = True,
    transform = transforms_apply, download = True
)

test_dataset = torchvision.datasets.MNIST(
    # root = './data', train = False,
    root = path_to_data + "data", train = False,
    transform = transforms_apply
)

# Create training and testing dataloaders-
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset, batch_size = batch_size,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset, batch_size = batch_size,
    shuffle = False
)


"""
Implementation in PyTorch

Currently, PyTorch doesn’t include an implementation of an STE in its APIs.
So, we will have to implement it ourselves. To do this we will need to create
a 'Function' class and a 'Module' class. The 'Function' class will contain the
'forward' and 'backward' functionality of the STE. The 'Module' class is where
the STE Function object will be created and used. We will use the STE Module
in our neural networks.
"""

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()


    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


"""
PyTorch lets us define custom autograd functions with forward and backward
functionality. Here we have defined an autograd function for a straight-through
estimator.

In the forward() method/pass we want to convert all the values in the input
tensor from floating point to binary (1/0).

In the backward() method/pass we want to pass the incoming gradients without
modifying them. This is to mimic the identity function. Although, here we are
performing 'F.hardtanh' operation on the incoming gradients. This operation
will clamp the gradient between -1 and 1. We are doing this so that the
gradients do not get too big.

Now, lets implement the STE Module class-
"""

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return STEFunction.apply(x)


"""
You can see that we have used the 'STEFunction' class we defined in the
forward() method. To use autograd functions we have to pass the input to the
apply() method. Now, we can use this module in our neural networks.

A common way to use STE is inside the bottleneck layer of autoencoders.
"""


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            StraightThroughEstimator(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh(),
        )


    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
        


model = Autoencoder().to(device)

"""
x = torch.randn((128, 1, 28, 28))
out, z = model(x)

# Sanity check-
# Since the STE at the end of the encoder, it will convert all of the
# continuous values of the tensor it receives to binary (1/0).

vals, cnts = z.unique(return_counts = True)

vals, cnts
# (tensor([0., 1.], grad_fn=<Unique2Backward0>), tensor([32768, 32768]))
"""

# Define gradient descent optimizer-
optimizer = torch.optim.Adam(
    params = model.parameters(),
    lr = learning_rate, weight_decay = 5e-4
)


def train_one_epoch(
    model, dataloader,
    train_dataset
):
    
    # Place model to device-
    model.to(device)
    
    # Enable training mode-
    model.train()
    
    # Initialize variables to reconstruction loss-
    running_recon_loss = 0.0
    
    for i, data in tqdm(
        enumerate(dataloader),
        total = int(len(train_dataset) / dataloader.batch_size)
        ):

        x, _ = data
        
        # Push data samples to device-
        x = x.to(device)
        
        # Empty accumulated gradients-
        optimizer.zero_grad()

        # Get latent 'z' and reconstructed input data-
        x_recon, _ = model(x)

        # Compute AE's reconstruction/task loss-
        recon_loss = F.mse_loss(input = x, target = x_recon)

        # Compute gradienst wrt computed loss-
        recon_loss.backward()
        
        # Perform one step of gradient descent-
        optimizer.step()
        
        # Compute total reconstruction loss-
        running_recon_loss += recon_loss.item()

        del x, x_recon, recon_loss
        
    # Compute loss as float value-
    recon_loss = running_recon_loss / len(dataloader.dataset)

    return recon_loss


def test_one_epoch(model, dataloader, test_dataset):
    
    # Place model to device-
    model.to(device)
    
    # Enable evaluation mode-
    model.eval()
    
    running_recon_loss = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader),
            total = int(len(test_dataset) / test_loader.batch_size)
        ):
            x, _ = data
        
            # Push data points to 'device'-
            x = x.to(device)
            
            # Forward propagation-
            x_recon, _ = model(x)
            
            # Compute reconstruction loss-
            recon_loss = F.mse_loss(input = x, target = x_recon)
            
            # Compute total reconstruction loss-
            running_recon_loss += recon_loss.item()

            del x, x_recon, recon_loss
                
    test_recon_loss = running_recon_loss / len(dataloader.dataset)
    
    return test_recon_loss


# Python dict to contain training metrics-
train_history = {}

# To save 'best' parameters-
best_test_loss = 10000


for epoch in range(1, num_epochs + 1):
    
    # Train model for 1 epoch-
    recon_train_loss = train_one_epoch(
        model = model, dataloader = train_loader,
        train_dataset = train_dataset
    )

    recon_test_loss = test_one_epoch(
        model = model, dataloader = test_loader,
        test_dataset = test_dataset
    )
    
    # Store model performance metrics in Python3 dict-
    train_history[epoch] = {
        'train_recon_loss': recon_train_loss,
        'test_recon_loss': recon_test_loss
    }
    
    print(f"Epoch = {epoch}; train MSE loss = {recon_train_loss:.6f}, ",
          f"& test MSE loss = {recon_test_loss:.6f}"
         )

    if recon_test_loss < best_test_loss:
        best_test_loss = recon_test_loss
        print(f"Saving model with lowest test_loss = {recon_test_loss:.6f}\n")
        
        # Save trained model with 'best' validation accuracy-
        torch.save(model.state_dict(), "ConvAE_STE_MNIST_best_model.pth")


with open("ConvAE_STE_MNIST_train_history.pkl", "wb") as file:
    pickle.dump(train_history, file)
del file


