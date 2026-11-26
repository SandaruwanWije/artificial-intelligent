# Libraries

Nobody writes backpropagation by hand for every new network. A deep learning
library takes that work over. This idea builds the same small network with the
two best known libraries, PyTorch and Keras, and checks each against the
`arrays` version. The numbers stay the same. What changes is how much is
written.

| Who writes it | Plain loops | `arrays` | PyTorch | Keras |
|---------------|-------------|----------|---------|-------|
| The layers | you | you | the library | the library |
| The loss | you | you | the library | the library |
| The way back, every slope | you | you | the library | the library |
| The update | you | you | the library | the library |
| The loop | you | you | you | the library |

| Class | What it is |
|-------|------------|
| `TorchNetwork(inputs, hidden)` | The network from PyTorch's own layers: `nn.Linear`, ReLU, `nn.Linear`, sigmoid. `start_from(array_network)` copies an `ArrayNetwork`'s numbers into it |
| `TorchSlopes(loss)` | `slopes(network, inputs, truths)` lets PyTorch find every slope, and gives them back as the `LayerSlopes` the arrays use |
| `TorchTrainer(network, learning_rate)` | The loop: `step` is five lines, and `train(inputs, truths, epochs)` runs it |
| `KerasNetwork(inputs, hidden, learning_rate)` | The same network as a Keras model. `start_from`, `answer`, and `train`, which is one call to `fit` |

PyTorch finds the slopes by automatic differentiation. It records every step
of the forward pass, and `loss.backward()` plays that record backwards,
multiplying each step's own slope into the next. That is the chain rule, so
the slopes are exact, not estimates.

Two details to know when comparing:

- **PyTorch keeps one row for each neuron.** Its hidden weights are (3, 2)
  where the arrays keep (2, 3). `start_from` and `TorchSlopes` turn them. Keras
  keeps one column for each neuron, like the arrays.
- **Both libraries use 32 bit numbers unless asked.** `TorchNetwork` asks for
  64 bit numbers, so it matches the arrays to about 15 decimals. `KerasNetwork`
  keeps Keras's own 32 bit numbers, and matches to about 7 decimals.

```python
from neural_network.libraries import KerasNetwork, TorchNetwork, TorchTrainer

network = TorchNetwork(inputs=2, hidden=3)
network.start_from(arrays)                              # an ArrayNetwork with the same shape
losses = TorchTrainer(network, learning_rate=0.5).train(mangoes.inputs, mangoes.truths, epochs=200)
losses[200]                                             # 0.3323, the same as ArrayTrainer

keras_network = KerasNetwork(inputs=2, hidden=3, learning_rate=0.5)
keras_network.start_from(arrays)
keras_network.train(mangoes.inputs, mangoes.truths, epochs=200)[200]   # 0.3323 again
```

PyTorch and Keras are in the `libraries` dependency group, which `uv` installs
by default. They are large. To leave them out, run `uv sync --no-group
libraries`, and this idea's tests are skipped. Keras does its sums with another
library underneath, and `keras_network.py` picks PyTorch for that.

`tests/test_libraries.py` checks the answers, the slopes and 200 epochs of
training against the arrays.

Uses: `arrays`.

Article: [Same network in PyTorch and Keras](https://byteblueprints.com/artificial-intelligence/ann/20_pytorch_ann/).

## See it work

```bash
uv run pytest tests/test_libraries.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_libraries.py::test_the_slopes_pytorch_finds_are_the_slopes_worked_out_by_hand PASSED
      hidden weight slopes, row 1: by hand [-0.0969, 0.0541, -0.0292]   PyTorch [-0.0969, 0.0541, -0.0292]
      output weight slopes:        by hand [-0.0363, -0.0553, 0.0298]   PyTorch [-0.0363, -0.0553, 0.0298]

tests/test_libraries.py::test_200_epochs_in_pytorch_follow_the_arrays_loss_at_every_step PASSED
      after epoch 0   arrays 0.6966   PyTorch 0.6966
      after epoch 1   arrays 0.6802   PyTorch 0.6802
      after epoch 50   arrays 0.4531   PyTorch 0.4531
      after epoch 100   arrays 0.3691   PyTorch 0.3691
      after epoch 200   arrays 0.3323   PyTorch 0.3323
      the furthest the two losses are ever apart 1.7e-16
```
