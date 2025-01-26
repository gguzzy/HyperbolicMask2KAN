import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# Define the models (WavKAN, LeviKAN, MLP) as provided earlier
# WavKAN and WavKANLinear
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WavKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(1, in_features))
        self.translation = nn.Parameter(torch.zeros(1, in_features))

        # Wavelet weights
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))

        # Normalization
        self.norm = nn.LayerNorm(out_features)

    def wavelet_transform(self, x):
        x_scaled = (x - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2 - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        else:
            # Implement other wavelet types if needed
            pass

        wavelet_output = wavelet @ self.wavelet_weights.T
        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        return self.norm(wavelet_output)

class WavKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(WavKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(WavKANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# LeviKAN and LeviCivitaKANLayer
class LeviCivitaKANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LeviCivitaKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters for scaling and translation
        self.scale = nn.Parameter(torch.ones(1, 1, in_features))
        self.translation = nn.Parameter(torch.zeros(1, 1, in_features))

        # Levi-Civita weights
        self.levicivita_weights = nn.Parameter(torch.Tensor(out_features, in_features * (in_features - 1)))
        nn.init.kaiming_uniform_(self.levicivita_weights, a=math.sqrt(5))

        # Base linear weights
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Normalization layer
        self.norm = nn.LayerNorm(out_features)

    def levicivita_transform(self, x):
        # x: (batch_size, in_features)
        batch_size, in_features = x.size()

        # Apply scaling and translation
        x_scaled = (x - self.translation.squeeze(1)) / self.scale.squeeze(1)  # (batch_size, in_features)

        # Compute pairwise products (excluding diagonal)
        x_expanded = x_scaled.unsqueeze(2)  # (batch_size, in_features, 1)
        x_transposed = x_scaled.unsqueeze(1)  # (batch_size, 1, in_features)
        pairwise_products = x_expanded * x_transposed  # (batch_size, in_features, in_features)

        # Exclude diagonal elements
        mask = ~torch.eye(in_features, dtype=bool, device=x.device).unsqueeze(0)
        pairwise_products = pairwise_products.masked_select(mask).view(batch_size, -1)

        # Compute output using matrix multiplication
        levicivita_output = pairwise_products @ self.levicivita_weights.T

        return levicivita_output

    def forward(self, x):
        # x: (batch_size, in_features)
        levicivita_output = self.levicivita_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output = levicivita_output + base_output
        return self.norm(combined_output)

class LeviKAN(nn.Module):
    def __init__(self, layers_hidden):
        super(LeviKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(LeviCivitaKANLayer(in_features, out_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

import random

class LeviCivitaKANLayerOptimized(nn.Module):
    def __init__(self, in_features, out_features, num_pairs=1000):
        super(LeviCivitaKANLayerOptimized, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pairs = num_pairs

        # Parameters for scaling and translation
        self.scale = nn.Parameter(torch.ones(1, in_features))
        self.translation = nn.Parameter(torch.zeros(1, in_features))

        # Indices of selected pairs
        self.register_buffer('indices_i', None)
        self.register_buffer('indices_j', None)
        self._select_pair_indices()

        # Levi-Civita weights
        self.levicivita_weights = nn.Parameter(torch.Tensor(out_features, num_pairs))
        nn.init.kaiming_uniform_(self.levicivita_weights, a=math.sqrt(5))

        # Base linear weights
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Normalization layer
        self.norm = nn.LayerNorm(out_features)

    def _select_pair_indices(self):
        indices = [(i, j) for i in range(self.in_features) for j in range(self.in_features) if i != j]
        selected_pairs = random.sample(indices, self.num_pairs)
        indices_i = torch.tensor([i for i, j in selected_pairs], dtype=torch.long)
        indices_j = torch.tensor([j for i, j in selected_pairs], dtype=torch.long)
        self.register_buffer('indices_i', indices_i)
        self.register_buffer('indices_j', indices_j)

    def levicivita_transform(self, x):
        # x: (batch_size, in_features)
        x_scaled = (x - self.translation) / self.scale  # (batch_size, in_features)

        # Compute selected pairwise products
        pairwise_products = x_scaled[:, self.indices_i] * x_scaled[:, self.indices_j]  # (batch_size, num_pairs)

        # Compute output using matrix multiplication
        levicivita_output = pairwise_products @ self.levicivita_weights.T  # (batch_size, out_features)

        return levicivita_output

    def forward(self, x):
        levicivita_output = self.levicivita_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output = levicivita_output + base_output
        return self.norm(combined_output)

class LeviKANOptimized(nn.Module):
    def __init__(self, layers_hidden, num_pairs=1000):
        super(LeviKANOptimized, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(LeviCivitaKANLayerOptimized(in_features, out_features, num_pairs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# MLP model
class MLP(nn.Module):
    """Very simple multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            layers.append(nn.Linear(n, k))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# [Include the model definitions from earlier here]
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Flatten the input images
        data = data.view(data.size(0), -1)  # Shape: (batch_size, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Optionally print training status
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Flatten the input images
            data = data.view(data.size(0), -1)  # Shape: (batch_size, 784)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')
    return test_loss, accuracy

# Prepare the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 64

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Model hyperparameters
input_dim = 28 * 28
hidden_dim = 256
output_dim = 10
num_layers = 3
layers_hidden = [input_dim, hidden_dim, output_dim]

# Initialize models
# Initialize models
wavkan_model = WavKAN(layers_hidden, wavelet_type='mexican_hat').to(device)
levikan_model = LeviKAN(layers_hidden).to(device)
mlp_model = MLP(input_dim, hidden_dim, output_dim, num_layers).to(device)
num_pairs = 1000  # Adjust as needed
levikanopt_model = LeviKANOptimized(layers_hidden, num_pairs).to(device)


# Add these lines
print('WavKAN Model Architecture:')
print(wavkan_model)
print('Number of trainable parameters in WavKAN:', count_parameters(wavkan_model))
# Add these lines
print('LeviKAN Model Architecture:')
print(levikan_model)
print('Number of trainable parameters in LeviKAN:', count_parameters(levikan_model))
print('LeviKAN OPTMIZED Model Architecture:')
print(levikanopt_model)
print('Number of trainable parameters in LeviKAN OPT:', count_parameters(levikanopt_model))
# Add these lines
print('MLP Model Architecture:')
print(mlp_model)
print('Number of trainable parameters in MLP:', count_parameters(mlp_model))
# Optimizers
learning_rate = 0.001
wavkan_optimizer = optim.Adam(wavkan_model.parameters(), lr=learning_rate)
levikan_optimizer = optim.Adam(levikan_model.parameters(), lr=learning_rate)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

# Training parameters
epochs = 1

# Training and evaluation functions (train and test) as defined earlier
# [Include the train and test function definitions here]

# Training and evaluating WavKAN
print('Training WavKAN Model')
for epoch in range(1, epochs + 1):
    train(wavkan_model, device, train_loader, wavkan_optimizer, epoch)
    test_loss, accuracy = test(wavkan_model, device, test_loader)

# Training and evaluating LeviKAN
print('Training LeviKAN Model')
for epoch in range(1, epochs + 1):
    train(levikan_model, device, train_loader, levikan_optimizer, epoch)
    test_loss, accuracy = test(levikan_model, device, test_loader)

print('Training LeviKAN OPTIMIZED Model')
for epoch in range(1, epochs + 1):
    train(levikanopt_model, device, train_loader, levikan_optimizer, epoch)
    test_loss, accuracy = test(levikanopt_model, device, test_loader)

# Training and evaluating MLP
print('Training MLP Model')
for epoch in range(1, epochs + 1):
    train(mlp_model, device, train_loader, mlp_optimizer, epoch)
    test_loss, accuracy = test(mlp_model, device, test_loader)
