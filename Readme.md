# Micrograd

Building a tiny autograd engine and neural network from scratch, following Andrej Karpathy's micrograd.

## Concepts

- **Derivatives**: what a derivative is
- **The `Value` object**: wrapping a number so it can track gradients
- **Operations**: implementing addition and multiplication
- **Derivative rules**: the local derivative of each operation
- **Chain rule**: combining local derivatives to get the gradient of each node with respect to the output

### What are neural nets?

Neural nets are mathematical expressions that take data as input.

## Getting started

1. Implement the `Value` class in [`base.py`](./base.py) and try the basic partial-derivative example there.
2. Use [`graph.py`](./graph.py) to draw computation-graph diagrams.
3. Build a single neuron's forward pass and its gradient in [`neural-network-example-micrograd.py`](./neural-network-example-micrograd.py).

## Files

| File | What it does |
|---|---|
| [`base.py`](./base.py) | `Value` class with operations and backprop |
| [`graph.py`](./graph.py) | `draw_dot` for visualizing the computation graph |
| [`neural-network-example-micrograd.py`](./neural-network-example-micrograd.py) | Single-neuron forward pass and gradient |
| [`neural_network.py`](./neural_network.py) | `Neuron`, `Layer` and `MLP` classes |
| [`tinydatasets.py`](./tinydatasets.py) | Tiny dataset, predictions and loss |
| [`pytorch-intro.py`](./pytorch-intro.py) | Same ideas in PyTorch |
