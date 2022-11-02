from FederatedModel import FederatedModel
from torch import nn
import torch.nn.functional as F
import itertools

# Configure your model architectures here
# Make sure that input size is num_features, and output size is num_labels
# These are supplied to the constructor as first and second parameters

# A basic linear model example.
class LinearModel(FederatedModel):
    def __init__(self, num_features, num_labels, args):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        return x


# A neural network with single hidden layer
class SingleLayer(FederatedModel):
    def __init__(self, num_features, num_labels, args):
        super().__init__()
        n = args.getint("neuron count")
        self.fc1 = nn.Linear(num_features, n)
        self.fc2 = nn.Linear(n, num_labels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
