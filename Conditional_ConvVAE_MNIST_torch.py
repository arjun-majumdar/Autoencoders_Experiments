

import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim

from tqdm import tqdm
from tqdm import trange
import os, pickle
import numpy as np
import matplotlib.pyplot as plt


"""
Refer-

https://github.com/arjun-majumdar/Autoencoders_Experiments/blob/master/Conditional_VAE_MNIST-PyTorch.ipynb
https://github.com/debtanu177/CVAE_MNIST/blob/master/train_cvae.py
https://pyro.ai/examples/cvae.html
"""


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


path_to_data = "C:\\Users\\demajuma\\Downloads\\"
batch_size = 512
num_epochs = 50


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

# Load MNIST dataset-
train_dataset = torchvision.datasets.MNIST(
    # root = './data', train = True,
    root = path_to_data, train = True,
    transform = transforms_apply, download = True
)

test_dataset = torchvision.datasets.MNIST(
    # root = './data', train = False,
    root = path_to_data, train = False,
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

'''
print(f"Sizes of train_dataset: {len(train_dataset)} and test_dataet: {len(test_dataset)}")
print(f"Sizes of train_loader: {len(train_loader)} and test_loader: {len(test_loader)}")
# Sizes of train_dataset: 60000 and test_dataet: 10000
# Sizes of train_loader: 118 and test_loader: 20

print(f"len(train_loader) = {len(train_loader)} & len(test_loader) = {len(test_loader)}")
# len(train_loader) = 118 & len(test_loader) = 20

# Sanity check-
len(train_dataset) / batch_size, len(test_dataset) / batch_size
# (117.1875, 19.53125)
'''


# Define Conditional Conv-VAE architecture:

class Encoder(nn.Module):
    def __init__(
        self, latent_dim = 5,
        num_classes = 10
    ):
        super().__init__()
    
        self.conv1 = nn.Conv2d(
            in_channels = 2, out_channels = 16,
            kernel_size = 5, stride = 2,
            bias = True
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, out_channels = 32,
            kernel_size = 5, stride = 2
        )
        self.linear = nn.Linear(in_features = 4 * 4 * 32, out_features = 300)
    
    
    def forward(self, x, y):
        label = np.zeros((x.size(0), 10))
        label[np.arange(x.shape[0]), y] = 1
        label = torch.from_numpy(label)

        y_t = torch.argmax(label, dim = 1).reshape((label.shape[0], 1, 1, 1))
        y_t = (torch.ones(x.shape) * y_t)
        x_t = torch.cat((x, y_t), dim = 1)

        out = F.leaky_relu(self.conv1(x_t))
        out = F.leaky_relu(self.conv2(out))
        return F.leaky_relu(self.linear(out.view(out.size(0), -1)))


class Decoder(nn.Module):
    def __init__(
        self, latent_dim = 5,
        num_classes = 10
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features = latent_dim + num_classes, out_features = 300)
        self.linear2 = nn.Linear(in_features = 300, out_features = 4 * 4 * 32)
        self.conv = nn.ConvTranspose2d(32, 16, kernel_size = 5,stride = 2)
        self.conv2 = nn.ConvTranspose2d(16, 1, kernel_size = 5, stride = 2)
        self.conv3 = nn.ConvTranspose2d(1, 1, kernel_size = 4)
        
        
    def forward(self, z, y):
        label = np.zeros((y.size(0), 10))

        label[np.arange(y.shape[0]), y] = 1
        label = torch.from_numpy(label)
        
        z = torch.cat((z, label.float()), dim = 1)

        out = F.leaky_relu(self.linear(z))
        out = F.leaky_relu(self.linear2(out))

        # Reshape before inputting to Conv2dtranspose-
        out = out.view(out.size(0), 32, 4, 4)

        out = F.leaky_relu(self.conv(out))
        out = F.leaky_relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))

        return out


class ConditionalVAE(nn.Module):
    def __init__(
        self, latent_dim = 5,
        num_classes = 10
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = Encoder(latent_dim = self.latent_dim)
        self.decoder = Decoder(latent_dim = self.latent_dim)
        
        self.mu = nn.Linear(in_features = 300, out_features = self.latent_dim)
        self.logvar = nn.Linear(in_features = 300, out_features = self.latent_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return (eps * std) + mu


    def forward(self, x, y):
    
        # Get embedded representation for input-
        out = self.encoder(x, y)
        
        # Get mean and log-var embeddings for encoded input-
        mu_embed = self.mu(out)
        logvar_embed = self.logvar(out)
        
        # Get reparameterized 'z'-
        z = self.reparameterize(mu = mu_embed, logvar = logvar_embed)
        
        # Synthesize a new output-
        x_synth = self.decoder(z, y)
        
        return x_synth, mu_embed, logvar_embed


latent_dim = 5
num_classes = 10

# Initialize Conditional Conv-VAE model-
model = ConditionalVAE(latent_dim = latent_dim, num_classes = num_classes).to(device)


def count_params(model):
    # Count number of layer-wise parameters and total parameters-
    tot_params = 0
    for param in model.parameters():
        # print(f"layer.shape = {param.shape} has {param.nelement()} parameters")
        tot_params += param.nelement()
    
    return tot_params

print(f"\nConditional Conv-VAE model has {count_params(model = model)} params\n\n")
# Conditional Conv-VAE model has 342704 params


# Define gradient descent optimizer-
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def compute_loss(data, data_recon, mu, log_var, alpha = 1, beta = 1):
    '''
    Function to compute loss = reconstruction loss * reconstruction_term_weight + KL-Divergence loss.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Inputs:
    1. mu: mean from the latent vector
    2. logvar: log variance from the latent vector
    3. alpha (int): Hyperparameter to control the importance of reconstruction
    loss vs KL-Divergence Loss - reconstruction term weight
    4. data: training data
    5. data_recon: VAE's reconstructed data
    '''
    
    # Sum over latent dimensions-
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), axis = 1)
    # kl_div = -0.5 * torch.sum(1 + log_var - (mu **2) - torch.exp(log_var), axis = 1)
    
    # kl_div.shape
    # torch.Size([32])
    
    batchsize = kl_div.size(0)

    # Average over batch dimension-
    kl_div = kl_div.mean()
    
    reconstruction_loss_fn = F.mse_loss
    recon_loss = reconstruction_loss_fn(data_recon, data, reduction = 'none')
    
    # recon_loss.shape
    # torch.Size([32, 1, 28, 28])
    
    # Sum over all pixels-
    recon_loss = recon_loss.view(batchsize, -1).sum(axis = 1)
    
    # recon_loss.shape
    # torch.Size([32])
    
    # Average over mini-batch dimension-
    recon_loss = recon_loss.mean()
    
    final_loss = (alpha * recon_loss) + (beta * kl_div)
    
    return final_loss, recon_loss, kl_div

'''
# Alternative cost computation-
recon_loss = F.mse_loss(x_synth, x, reduction = 'sum')
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
'''


def train_one_epoch(model, dataloader, dataset, alpha, beta):
    
    # Place model to device-
    model.to(device)
    
    # Enable training mode-
    model.train()
    
    # Initialize variables to keep track of 3 losses-
    running_final_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    
    
    for i, data in tqdm(
        enumerate(dataloader),
        total = int(len(dataset) / dataloader.batch_size)
        ):
      
        x = data[0]
        y = data[1]
        
        # Push to 'device'-
        x = x.to(device)
        y = y.to(device)
        
        # Empty accumulated gradients-
        optimizer.zero_grad()
        
        # Perform forward propagation-
        x_recon, mu, logvar = model(x, y)
        
        final_loss, recon_loss, kl_div_loss = compute_loss(
            data = x, data_recon = x_recon,
            mu = mu, log_var = logvar,
            alpha = alpha, beta = beta
        )
        
        # Update losses-
        running_final_loss += final_loss.item()
        running_kl_loss += kl_div_loss.cpu().detach().numpy()
        running_recon_loss += recon_loss.cpu().detach().numpy()
        
        # Compute gradients wrt total loss-
        final_loss.backward()
        
        # Perform gradient descent-
        optimizer.step()
    
    # Compute losses as float values-
    train_loss = running_final_loss / len(dataloader.dataset)
    kl_loss = running_kl_loss / len(dataloader.dataset)
    recon_loss = running_recon_loss / len(dataloader.dataset)
    
    return train_loss, kl_loss, recon_loss


def test_one_epoch(model, dataloader, dataset, alpha, beta):
    
    # Place model to device-
    model.to(device)
    
    # Enable evaluation mode-
    model.eval()
    
    running_final_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader),
            total = int(len(dataset) / dataloader.batch_size)
        ):
            
            x_v = data[0]
            y_v = data[1]
            
            # Push data points to 'device'-
            x_v = x_v.to(device)
            y_v = y_v.to(device)
            
            # Forward propagation-
            x_recon, mu, logvar = model(x_v, y_v)
            
            final_loss, recon_loss, kl_div_loss = compute_loss(
                data = x_v, data_recon = x_recon,
                mu = mu, log_var = logvar,
                alpha = alpha, beta = beta
            )
        
            # Update losses-
            running_final_loss += final_loss.item()
            running_kl_loss += kl_div_loss.cpu().detach().numpy()
            running_recon_loss += recon_loss.cpu().detach().numpy()
            
                
    test_loss = running_final_loss / len(dataloader.dataset)
    test_kl_loss = running_kl_loss / len(dataloader.dataset)
    test_recon_loss = running_recon_loss / len(dataloader.dataset)
    
    return test_loss, test_kl_loss, test_recon_loss


# Specify alpha - Hyperparameter to control the importance of reconstruction
# loss vs KL-Divergence Loss-
alpha = 1
beta = 1


# Python dict to contain training metrics-
train_history = {}

# Initialize parameters for Early Stopping manual implementation-
best_test_loss = 10000


for epoch in range(1, num_epochs + 1):
    
    train_loss, kl_train_loss, recon_train_loss = train_one_epoch(
        model = model, dataloader = train_loader,
        dataset = train_dataset,
        alpha = alpha, beta = beta
    )
    
    test_loss, test_kl_loss, test_recon_loss = test_one_epoch(
        model = model, dataloader = test_loader,
        dataset = test_dataset, 
        alpha = alpha, beta = beta
    )
    
    # Store model performance metrics in Python3 dict-
    train_history[epoch] = {
        'train_loss': train_loss,
        'train_recon_loss': kl_train_loss,
        'train_kl_loss': kl_train_loss,
        'test_loss': test_loss,
        'test_recon_loss': test_recon_loss,
        'test_kl_loss': test_kl_loss
    }
    
    print(f"Epoch = {epoch}; loss = {train_loss:.4f}",
          f", kl-loss = {kl_train_loss:.4f}, recon loss = {recon_train_loss:.4f}",
          f", test loss = {test_loss:.4f}, test kl-loss = {test_kl_loss:.4f}",
          f" & test recon loss = {test_recon_loss:.4f}"
         )
    
    
    # Code for manual Early Stopping:
    if (test_loss < best_test_loss):
    
        # update variable to lowest loss encountered so far-
        best_test_loss = test_loss
        
        print(f"Saving model with lowest val_loss = {test_loss:.4f}\n")
        torch.save(model.state_dict(), "Conditional_ConvVAE_MNIST_best_model.pth")


# Save training history as pickle file-
with open("Conditional_ConvVAE_MNIST_train_history.pkl", "wb") as file:
    pickle.dump(train_history, file)


# Train Visualizations:

plt.figure(figsize = (10, 8))
plt.plot([train_history[e]['train_loss'] for e in train_history.keys()], label = 'loss')
plt.plot([train_history[e]['test_loss'] for e in train_history.keys()], label = 'val loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Conv-VAE train loss visualization")
plt.legend(loc = 'best')
plt.show()


plt.figure(figsize = (10, 8))
plt.plot([train_history[e]['train_kl_loss'] for e in train_history.keys()], label = 'loss')
plt.plot([train_history[e]['test_kl_loss'] for e in train_history.keys()], label = 'val loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Conv-VAE train kl-loss visualization")
plt.legend(loc = 'best')
plt.show()


plt.figure(figsize = (10, 8))
plt.plot([train_history[e]['train_recon_loss'] for e in train_history.keys()], label = 'loss')
plt.plot([train_history[e]['test_recon_loss'] for e in train_history.keys()], label = 'val loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Conv-VAE train recon visualization")
plt.legend(loc = 'best')
plt.show()


del model

 

# Load trained weights from before-
# trained_model = ConditionalVAE(latent_dim = latent_dim, num_classes = num_classes).to(device)
# trained_model.load_state_dict(torch.load('Conditional_ConvVAE_MNIST_best_model.pth', map_location = torch.device('cpu')))


def synthesize_images(model, label = 2, batch_size = 128, latent_dim = 5):
    # Synthesize new MNIST images using trained 'model'.
    
    z = torch.randn(batch_size, latent_dim)
    # labels = F.one_hot(torch.randint(0, 9, (128,)), num_classes = 10)
    y = torch.from_numpy(np.asarray([label] * batch_size))
    
    # Synthesize a new output-
    x_synth = model.decoder(z, y).detach().cpu().numpy()
    
    return np.transpose(a = x_synth, axes = (0, 2, 3, 1))

# Specify class/label to generate digits-
label = 7

# Get synthesized digits-
x_synth = synthesize_images(
    model = model, label = label,
    batch_size = batch_size, latent_dim = latent_dim
)

# Visualize synthesized images using trained Conditional Conv-VAE-
plt.figure(figsize = (12, 10))
for i in range(50):
    # 10 rows & 5 columns-
    plt.subplot(10, 5, i + 1)
    plt.imshow(x_synth[i], cmap = 'gray')
    plt.axis('off')
    
plt.suptitle(f"Synthesized MNIST {label} images")
plt.show()


