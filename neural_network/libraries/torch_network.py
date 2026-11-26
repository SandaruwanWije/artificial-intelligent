"""The same network in PyTorch: a layer is an object, and PyTorch finds the slopes."""

import numpy as np
import torch
from torch import nn

from neural_network.arrays import ArrayNetwork, LayerSlopes


class TorchNetwork(nn.Module):
    """Some inputs, a hidden layer of ReLU neurons and one sigmoid answer, built from PyTorch's own layers."""

    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden)
        self.output = nn.Linear(hidden, 1)
        # PyTorch uses 32 bit numbers unless asked. With 64 bit numbers it matches the arrays digit for digit.
        self.double()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden(inputs))
        return torch.sigmoid(self.output(hidden))

    def start_from(self, network: ArrayNetwork) -> None:
        """Copies an array network's weights and biases, so both networks start from the same numbers."""
        layers = [self.hidden, self.output]
        for layer_number in range(len(layers)):
            ours = network.layers[layer_number]
            # PyTorch keeps one row for each neuron. The arrays keep one column, so they are turned on their side.
            layers[layer_number].weight.data = torch.tensor(ours.weights.T.copy())
            layers[layer_number].bias.data = torch.tensor(ours.biases.copy())


class TorchSlopes:
    """Every slope, found by PyTorch: it records the forward pass, then plays the record backwards."""

    def __init__(self, loss: nn.Module):
        self.loss = loss

    def slopes(self, network: TorchNetwork, inputs: np.ndarray, truths: np.ndarray) -> list[LayerSlopes]:
        """One LayerSlopes per layer, laid out like the arrays: one column for each neuron."""
        network.zero_grad()
        answers = network(torch.tensor(inputs))
        loss = self.loss(answers, torch.tensor(truths))
        # This one call walks back from the loss and leaves each slope beside the number it belongs to.
        loss.backward()
        all_slopes = []
        for layer in [network.hidden, network.output]:
            all_slopes.append(LayerSlopes(layer.weight.grad.T.numpy(), layer.bias.grad.numpy()))
        return all_slopes


class TorchTrainer:
    """The training loop in PyTorch: the loss and the optimizer are picked by name, and each step is one line."""

    def __init__(self, network: TorchNetwork, learning_rate: float):
        self.network = network
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    def step(self, inputs: torch.Tensor, truths: torch.Tensor) -> float:
        """One trip around the loop. Returns the loss as it was before the step."""
        self.optimizer.zero_grad()
        answers = self.network(inputs)
        loss = self.loss(answers, truths)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, inputs: np.ndarray, truths: np.ndarray, epochs: int) -> list[float]:
        """Trains for some epochs, every example in each step. Returns the loss before training and after each epoch."""
        inputs = torch.tensor(inputs)
        truths = torch.tensor(truths)
        losses = []
        for epoch in range(epochs):
            losses.append(self.step(inputs, truths))
        # Each step gave the loss before it moved the weights, so the loss after the last step is still missing.
        with torch.no_grad():
            last = self.loss(self.network(inputs), truths)
        losses.append(last.item())
        return losses
