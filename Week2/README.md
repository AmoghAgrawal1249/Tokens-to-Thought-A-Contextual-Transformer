# Week 2: Neural Networks from Scratch

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

This week was all about understanding how neural networks actually work under the hood. I implemented everything from scratch using only NumPy — no TensorFlow, no PyTorch, no sklearn. The goal was to build intuition for forward propagation, backpropagation, and gradient-based learning.

## Completed Assignments

All implementations are in `feedforward_from_scratch.py`.

### 1. Perceptron & AND Gate ✓

Built a single-neuron perceptron with:
- Step activation function
- Zero-initialized weights with optional bias
- Perceptron learning rule (weight update on misclassification)
- Early stopping when epoch completes error-free

**Result:** Successfully learned the AND gate truth table.

```
Input (0, 0) -> 0
Input (0, 1) -> 0
Input (1, 0) -> 0
Input (1, 1) -> 1
```

### 2. XOR Gate with Single Perceptron ✗

Attempted to learn XOR with the same perceptron architecture.

**Result:** Failed — as expected! XOR is not linearly separable, so a single perceptron cannot model it. This exercise demonstrated the fundamental limitation that motivated the development of multi-layer networks.

### 3. XOR Gate with Hidden Layer ✓

Implemented a two-layer neural network (`TwoLayerNetwork` class) with:
- Xavier/Glorot weight initialization
- Hidden layer with `tanh` activation
- Output layer with `sigmoid` activation
- Binary cross-entropy loss
- Fully vectorized backpropagation

**Result:** Perfect XOR reconstruction after ~5000 epochs.

### 4. Full Adder ✓

Extended the two-layer network to learn the full adder function:
- **Inputs:** bit_A, bit_B, carry_in (3 inputs)
- **Outputs:** sum, carry_out (2 outputs)

All 8 input combinations mapped correctly to their expected outputs.

### 5. Ripple-Carry Adder ✓

Composed the trained full adder to perform multi-bit addition:
- Processes bits from LSB to MSB
- Propagates carry between positions
- Successfully adds 4-bit numbers (e.g., 13 + 11 = 24)

## Implementation Details

### Network Architecture

```
Input Layer → Hidden Layer (tanh) → Output Layer (sigmoid)
     ↓              ↓                      ↓
   [n_in]    [hidden_units]           [n_out]
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `sigmoid()` | Activation for output layer |
| `tanh_derivative()` | Used in backprop through hidden layer |
| `initialize_parameters()` | Xavier-style weight init |
| `forward_propagation()` | Compute activations layer by layer |
| `compute_loss()` | Binary cross-entropy with numerical stability |
| `backward_propagation()` | Vectorized gradient computation |
| `update_parameters()` | Gradient descent step |

## How to Run

```bash
cd Week2
python feedforward_from_scratch.py
```

Expected output:
```
AND gate truth table using a single perceptron:
Input (0, 0) -> 0
Input (0, 1) -> 0
Input (1, 0) -> 0
Input (1, 1) -> 1

Attempting XOR with a single perceptron:
...
Result: A single perceptron fails to model the XOR gate due to non-linear separability.

XOR gate using a two-layer neural network:
Input (0, 0) -> 0
Input (0, 1) -> 1
Input (1, 0) -> 1
Input (1, 1) -> 0

Full adder (Sum, Carry) using a two-layer neural network:
...

Ripple carry addition using the learned full adder network:
Operands (LSB→MSB): [1 0 1 1] + [1 1 0 1]
Result bits (LSB→MSB): [0 0 1 1 0]
Decimal check: 13 + 11 = 24
```

## Key Learnings

1. **Linear separability** is the fundamental constraint of single-layer networks
2. **Hidden layers** create non-linear decision boundaries by composing multiple linear transformations with non-linear activations
3. **Vectorization** makes backpropagation tractable — matrix operations over entire batches at once
4. **Xavier initialization** prevents vanishing/exploding gradients at the start of training

## Files in This Directory

```
Week2/
├── feedforward_from_scratch.py   # All neural network implementations
└── README.md                     # This file
```

## Resources I Found Helpful

- [3Blue1Brown: Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- [3Blue1Brown: Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Michael Nielsen's Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

---

*Completed: Week 2 of Tokens-to-Thought course*
