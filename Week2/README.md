# Module 2: Neural Networks from First Principles

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

This module tackles the core question: **How do neural networks actually learn?** Rather than using TensorFlow or PyTorch, I implemented everything from scratch using only NumPy. This constraint forced me to understand every detail—weight initialization, forward propagation, loss computation, backpropagation, and gradient descent.

The deliverables include a working perceptron, a two-layer neural network, and practical applications: logic gates and a ripple-carry adder built entirely from learned neurons.

## Project Files

```
Week2/
├── feedforward_from_scratch.py    # Complete neural network implementation
└── README.md                      # This documentation
```

---

## Part 1: The Perceptron

### What is a Perceptron?

A perceptron is the simplest possible neural network—a single neuron that computes:

$$\hat{y} = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

For classification, I used the step activation:

$$\sigma(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

### Implementation

```python
class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, epochs=50):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def initialize(self):
        """Start with zero weights (could also use small random values)"""
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0
    
    def net_input(self, X):
        """Compute weighted sum: z = Wx + b"""
        return X @ self.weights + self.bias
    
    def predict(self, X):
        """Apply step activation"""
        return (self.net_input(X) >= 0).astype(int)
    
    def fit(self, X, y):
        """Train using the perceptron learning rule"""
        self.initialize()
        
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = 1 if (np.dot(xi, self.weights) + self.bias) >= 0 else 0
                error = target - prediction
                
                if error != 0:
                    # Update rule: w <- w + lr * error * x
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            
            # Early stopping if perfectly classified
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
```

### The Perceptron Learning Rule

When a sample is misclassified:
- If prediction = 0 but target = 1: error = +1, so weights increase toward the input
- If prediction = 1 but target = 0: error = -1, so weights decrease away from the input

This rule is guaranteed to converge for linearly separable data.

---

## Part 2: Logic Gates as Neural Networks

### AND Gate: Success ✓

The AND gate outputs 1 only when both inputs are 1:

| $x_1$ | $x_2$ | AND |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**Why it works:** The AND function is linearly separable. A line can divide the (1,1) point from the rest.

```python
# Training AND gate
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

perceptron = Perceptron(n_features=2, learning_rate=0.2, epochs=20)
perceptron.fit(data, targets)
print(perceptron.predict(data))  # [0, 0, 0, 1] ✓
```

The perceptron learns weights like $w_1 = 0.2, w_2 = 0.2, b = -0.3$, creating the decision boundary $0.2x_1 + 0.2x_2 - 0.3 = 0$.

### XOR Gate: Failure ✗

The XOR gate outputs 1 when inputs differ:

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Why it fails:** XOR is NOT linearly separable. No single line can separate the 1s from the 0s.

```
    x2
    1 |  (0,1)=1     (1,1)=0
      |      ●           ○
      |
    0 |  (0,0)=0     (1,0)=1
      |      ○           ●
      +---------------------- x1
           0            1
```

No matter how long training runs, the perceptron cannot solve XOR. This limitation motivated the development of multi-layer networks.

---

## Part 3: Two-Layer Neural Network

### Architecture

To solve non-linearly separable problems, I added a hidden layer:

```
Input Layer      Hidden Layer       Output Layer
    [2]      →      [n_h]       →      [1]
              (tanh)            (sigmoid)
```

### Mathematical Formulation

**Forward Propagation:**

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$
$$\mathbf{a}^{[1]} = \tanh(\mathbf{z}^{[1]})$$
$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$
$$\hat{y} = \sigma(\mathbf{z}^{[2]}) = \frac{1}{1 + e^{-\mathbf{z}^{[2]}}}$$

**Loss Function (Binary Cross-Entropy):**

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### Backpropagation Derivation

The key insight is the chain rule. Starting from the loss:

**Output Layer Gradients:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} = \hat{y} - y$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \frac{1}{m} (\mathbf{a}^{[1]})^T (\hat{y} - y)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} = \frac{1}{m} \sum (\hat{y} - y)$$

**Hidden Layer Gradients:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \left[(\hat{y} - y) (\mathbf{W}^{[2]})^T\right] \odot (1 - (\mathbf{a}^{[1]})^2)$$

The term $(1 - \mathbf{a}^2)$ is the derivative of tanh.

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \frac{1}{m} \mathbf{x}^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}}$$

### Implementation

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh_derivative(a):
    return 1.0 - a**2

def initialize_parameters(input_dim, hidden_dim, output_dim, rng):
    """Xavier/Glorot initialization for stable training"""
    limit_h = np.sqrt(6 / (input_dim + hidden_dim))
    limit_o = np.sqrt(6 / (hidden_dim + output_dim))
    
    return {
        "W1": rng.uniform(-limit_h, limit_h, (input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim)),
        "W2": rng.uniform(-limit_o, limit_o, (hidden_dim, output_dim)),
        "b2": np.zeros((1, output_dim)),
    }

def forward(X, params):
    """Forward pass through the network"""
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = sigmoid(z2)
    return {"A0": X, "Z1": z1, "A1": a1, "Z2": z2, "A2": a2}

def compute_loss(predictions, targets):
    """Binary cross-entropy with numerical stability"""
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    loss = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return np.mean(loss)

def backward(targets, cache, params):
    """Compute gradients via backpropagation"""
    m = targets.shape[0]
    
    # Output layer
    dz2 = cache["A2"] - targets
    dW2 = cache["A1"].T @ dz2 / m
    db2 = np.mean(dz2, axis=0, keepdims=True)
    
    # Hidden layer
    dz1 = (dz2 @ params["W2"].T) * tanh_derivative(cache["A1"])
    dW1 = cache["A0"].T @ dz1 / m
    db1 = np.mean(dz1, axis=0, keepdims=True)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(params, grads, learning_rate):
    """Gradient descent step"""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params
```

### XOR Gate: Now Solved ✓

With a hidden layer of 4 neurons, the network learns XOR:

```python
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

network = TwoLayerNetwork(hidden_units=4, learning_rate=0.6, epochs=5000, seed=42)
network.fit(data, targets)

print(network.predict(data))
# [[0], [1], [1], [0]] ✓
```

The hidden layer creates an intermediate representation where XOR becomes linearly separable.

---

## Part 4: Arithmetic Circuits

### Full Adder

A full adder takes three binary inputs (A, B, Carry_in) and produces two outputs (Sum, Carry_out):

| A | B | C_in | Sum | C_out |
|---|---|------|-----|-------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 0 | 1 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 | 0 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 | 1 |

I trained a network with 6 hidden units to learn this mapping:

```python
data = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
])
targets = np.array([
    [0, 0], [1, 0], [1, 0], [0, 1],
    [1, 0], [0, 1], [0, 1], [1, 1]
])

adder = TwoLayerNetwork(hidden_units=6, learning_rate=0.8, epochs=8000, seed=7)
adder.fit(data, targets)
```

**Result:** 100% accuracy on all 8 input combinations.

### Ripple-Carry Adder

By composing the learned full adder, I built a multi-bit adder:

```python
def ripple_add(a_bits, b_bits, full_adder_network):
    """Add two binary numbers using the learned full adder"""
    carry = 0
    result = []
    
    for bit_a, bit_b in zip(a_bits, b_bits):
        inputs = np.array([[bit_a, bit_b, carry]])
        sum_bit, carry = full_adder_network.predict(inputs)[0]
        result.append(sum_bit)
    
    result.append(carry)  # Final carry
    return np.array(result)

# Test: 13 + 11 = 24
a = np.array([1, 0, 1, 1])  # 13 in binary (LSB first)
b = np.array([1, 1, 0, 1])  # 11 in binary (LSB first)
result = ripple_add(a, b, adder)
# [0, 0, 0, 1, 1] = 24 in binary ✓
```

**Output:**
```
Ripple carry addition using the learned full adder network:
Operands (LSB→MSB): [1 0 1 1] + [1 1 0 1]
Result bits (LSB→MSB): [0 0 0 1 1]
Decimal check: 13 + 11 = 24
```

---

## Part 5: Key Implementation Details

### Why Xavier Initialization?

Random initialization scale matters. If weights are too large, activations saturate and gradients vanish. Xavier initialization sets:

$$W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

This keeps variance stable across layers.

### Why tanh for Hidden Layers?

- **Centered output:** tanh outputs in [-1, 1], keeping activations centered around 0
- **Stronger gradients:** Compared to sigmoid, tanh has steeper gradients in the linear region
- **Non-linearity:** Creates the non-linear decision boundaries needed for XOR

### Why sigmoid for Output?

For binary classification, we want outputs in [0, 1] interpretable as probabilities. Sigmoid provides this naturally.

### Numerical Stability

Cross-entropy loss can produce NaN if predictions are exactly 0 or 1:

```python
# Unsafe
loss = -y * np.log(pred) - (1-y) * np.log(1-pred)  # log(0) = -inf!

# Safe
pred = np.clip(pred, 1e-12, 1 - 1e-12)
loss = -y * np.log(pred) - (1-y) * np.log(1-pred)
```

---

## Running the Code

```bash
cd Week2
python feedforward_from_scratch.py
```

### Expected Output

```
AND gate truth table using a single perceptron:
Input (0, 0) -> 0
Input (0, 1) -> 0
Input (1, 0) -> 0
Input (1, 1) -> 1

Attempting XOR with a single perceptron:
Input (0, 0) -> predicted 0, target 0 ✓
Input (0, 1) -> predicted 0, target 1 ✗
Input (1, 0) -> predicted 0, target 1 ✗
Input (1, 1) -> predicted 0, target 0 ✓
Result: A single perceptron fails to model the XOR gate due to non-linear separability.

XOR gate using a two-layer neural network:
Input (0, 0) -> 0
Input (0, 1) -> 1
Input (1, 0) -> 1
Input (1, 1) -> 0

Full adder (Sum, Carry) using a two-layer neural network:
Input (0, 0, 0) -> Sum 0, Carry 0
Input (0, 0, 1) -> Sum 1, Carry 0
...

Ripple carry addition using the learned full adder network:
Operands (LSB→MSB): [1 0 1 1] + [1 1 0 1]
Result bits (LSB→MSB): [0 0 0 1 1]
Decimal check: 13 + 11 = 24
```

---

## Key Takeaways

1. **Single neurons have hard limits** — Linear separability is a real constraint, not a theoretical curiosity

2. **Hidden layers create features** — The network learns intermediate representations where problems become solvable

3. **Backpropagation is just calculus** — Chain rule applied systematically, nothing magical

4. **Vectorization enables scale** — Matrix operations make training millions of samples practical

5. **Initialization matters** — Xavier initialization prevents training from getting stuck

---

## References

- [3Blue1Brown: Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- [3Blue1Brown: Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Michael Nielsen: Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- Glorot & Bengio (2010): Understanding the difficulty of training deep feedforward neural networks

---

*This module proved that understanding beats memorization. Implementing from scratch makes frameworks feel like conveniences rather than mysteries.*
