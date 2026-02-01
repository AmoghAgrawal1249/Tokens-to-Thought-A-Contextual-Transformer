# Module 3: TensorFlow Deep Dive

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

After implementing neural networks from scratch in Module 2, this module explores TensorFlow—not as a black box, but as a tool whose internals I now understand. The goal was twofold: learn to use TensorFlow's high-level APIs efficiently, and demystify them by implementing custom layer classes that replicate built-in functionality.

I applied these skills to two classic problems: MNIST digit classification and housing price regression.

## Project Files

```
Week3/
├── Week3.ipynb     # Complete notebook with all implementations
└── README.md       # This documentation
```

---

## Part 1: MNIST Classification with Built-in Layers

### The Dataset

MNIST is the "Hello World" of machine learning—60,000 training images and 10,000 test images of handwritten digits (0-9), each 28×28 grayscale pixels.

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### Model Architecture

A simple feedforward network:

```
Input: 28×28 image (784 pixels when flattened)
    ↓
Flatten: 784 neurons
    ↓
Dense: 128 neurons, ReLU activation
    ↓
Dense: 10 neurons, Softmax activation
    ↓
Output: Probability distribution over digits 0-9
```

### Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
```

### Results

| Metric | Value |
|--------|-------|
| Training Accuracy (5 epochs) | 98.7% |
| Validation Accuracy | 97.5% |
| **Test Accuracy** | **97.2%** |

Achieving 97%+ accuracy with just 128 hidden neurons demonstrates the power of even simple neural networks on structured data.

---

## Part 2: Custom Layer Implementation

### Why Build Custom Layers?

TensorFlow's built-in `Dense` layer is convenient, but understanding what happens inside is crucial for:
- Debugging training issues
- Implementing novel architectures
- Optimizing for specific hardware

### The Layer Lifecycle

TensorFlow layers have three key methods:

1. **`__init__(self, ...)`**: Store configuration (units, activation type)
2. **`build(self, input_shape)`**: Create weights when input shape is known
3. **`call(self, inputs)`**: Define the forward pass computation

### CustomDenseReluLayer

```python
class CustomDenseReluLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomDenseReluLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Weight matrix: [input_dim, units]
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        # Bias vector: [units]
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Linear transformation: z = Wx + b
        z = tf.matmul(inputs, self.w) + self.b
        # ReLU activation: max(0, z)
        return tf.nn.relu(z)
```

**Key Points:**
- `add_weight()` registers trainable parameters with TensorFlow's gradient tape
- `glorot_uniform` (Xavier) initialization prevents vanishing gradients
- `tf.matmul` performs batched matrix multiplication

### CustomDenseSoftmaxLayer

```python
class CustomDenseSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomDenseSoftmaxLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        # Softmax: convert logits to probability distribution
        return tf.nn.softmax(z, axis=-1)
```

**Softmax Formula:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

This ensures outputs sum to 1 and can be interpreted as probabilities.

### CustomFlattenLayer

```python
class CustomFlattenLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Get batch size dynamically
        batch_size = tf.shape(inputs)[0]
        # Flatten all other dimensions
        return tf.reshape(inputs, (batch_size, -1))
```

The `-1` in reshape tells TensorFlow to infer the dimension from the total number of elements.

### Custom Model for MNIST

```python
custom_model = Sequential([
    CustomFlattenLayer(),
    CustomDenseReluLayer(128),
    CustomDenseSoftmaxLayer(10)
])

custom_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

custom_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

test_loss, test_acc = custom_model.evaluate(x_test, y_test)
print(f'Custom model test accuracy: {test_acc:.4f}')
```

### Verification

| Model | Test Accuracy |
|-------|---------------|
| Built-in layers | 97.2% |
| Custom layers | 97.1% |

The custom implementation achieves essentially identical performance, confirming correctness.

---

## Part 3: Housing Price Regression

### The Dataset

I used the California Housing dataset (a modern replacement for the deprecated Boston Housing dataset):

- **Samples:** 20,640 houses
- **Features:** 8 (median income, house age, average rooms, etc.)
- **Target:** Median house value (in $100,000s)

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling is crucial for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Why Scale Features?

Neural networks are sensitive to feature scales. Without normalization:
- Features with large values dominate the loss
- Gradient descent takes inefficient paths
- Training may not converge

StandardScaler transforms each feature to have mean=0 and std=1.

### Model 1: Linear Regression

A neural network with one neuron and no activation is equivalent to linear regression:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

```python
linear_model = Sequential([
    Dense(1, input_shape=(8,), activation=None)
])

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

linear_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
```

### Model 2: Feedforward Neural Network

A deeper network can capture non-linear relationships:

```python
feedforward_model = Sequential([
    Dense(64, input_shape=(8,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation=None)  # No activation for regression
])

feedforward_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

feedforward_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
```

### Results Comparison

| Model | Architecture | Test MSE | Test MAE |
|-------|-------------|----------|----------|
| Linear Regression | 8 → 1 | 0.524 | 0.533 |
| Feedforward NN | 8 → 64 → 32 → 16 → 1 | 0.261 | 0.347 |

**Key Findings:**
- The neural network achieves **50% lower MSE** than linear regression
- This improvement comes from learning non-linear interactions between features
- The MAE of 0.347 means predictions are off by ~$34,700 on average

### Training Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(history_linear.history['loss'], label='Linear - Train')
axes[0].plot(history_linear.history['val_loss'], label='Linear - Val')
axes[0].plot(history_ff.history['loss'], label='NN - Train')
axes[0].plot(history_ff.history['val_loss'], label='NN - Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()

# MAE curves
axes[1].plot(history_linear.history['mae'], label='Linear - Train')
axes[1].plot(history_ff.history['mae'], label='NN - Train')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
```

---

## Part 4: Key Concepts Explained

### Loss Functions

**Classification (Cross-Entropy):**
$$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$

Penalizes confident wrong predictions heavily.

**Regression (MSE):**
$$\mathcal{L} = \frac{1}{n} \sum_{i} (y_i - \hat{y}_i)^2$$

Penalizes large errors more than small ones (quadratic).

### The Adam Optimizer

Adam combines:
- **Momentum:** Accelerates convergence by accumulating gradient direction
- **RMSprop:** Adapts learning rate per-parameter based on gradient history

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,  # Base learning rate
    beta_1=0.9,           # Momentum decay
    beta_2=0.999          # RMSprop decay
)
```

### Validation Split

```python
model.fit(..., validation_split=0.2)
```

This holds out 20% of training data to monitor overfitting. If validation loss increases while training loss decreases, the model is memorizing rather than learning.

### Batch Size Tradeoffs

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (8-32) | More updates, better generalization | Noisy gradients, slower |
| Large (128-512) | Stable gradients, faster | May converge to sharp minima |

I used batch_size=32 as a reasonable default.

---

## Running the Code

### Requirements

```bash
pip install tensorflow scikit-learn matplotlib pandas numpy
```

### Execute Notebook

```bash
jupyter notebook Week3.ipynb
```

Or upload to Google Colab for GPU acceleration (recommended for faster training).

### Expected Output

```
MNIST with Built-in Layers:
Epoch 5/5 - accuracy: 0.9872 - val_accuracy: 0.9752
Test accuracy: 0.9720

MNIST with Custom Layers:
Epoch 5/5 - accuracy: 0.9865 - val_accuracy: 0.9748
Test accuracy: 0.9710

Housing Price Regression:
Linear Regression - Test MSE: 0.5240, Test MAE: 0.5330
Feedforward NN    - Test MSE: 0.2610, Test MAE: 0.3470
```

---

## Part 5: Lessons Learned

### 1. Frameworks Abstract Complexity, Not Magic

After implementing layers from scratch, TensorFlow's `Dense(128, activation='relu')` feels like syntactic sugar rather than a black box. I know exactly what computations happen inside.

### 2. Feature Scaling is Critical

The housing regression model wouldn't converge without StandardScaler. Always normalize inputs for neural networks.

### 3. Architecture Search is Empirical

I experimented with different hidden layer sizes:
- 128 → 64 → 32 → 1: MSE = 0.28
- 64 → 32 → 16 → 1: MSE = 0.26 (best)
- 256 → 128 → 64 → 1: MSE = 0.27 (overfitting)

Bigger isn't always better.

### 4. Validation Prevents Overfitting

Monitoring validation loss helped me choose when to stop training. Without it, I would have trained too long and overfit.

---

## Extensions Implemented

Beyond the core requirements, I added:

1. **Training visualization:** Side-by-side loss and MAE curves
2. **Model comparison:** Quantitative comparison table
3. **Custom layers:** Full implementations matching TensorFlow behavior

---

## References

- [TensorFlow Custom Layers Guide](https://www.tensorflow.org/tutorials/customization/custom_layers)
- [Keras Sequential Model](https://keras.io/guides/sequential_model/)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)

---

*This module bridged the gap between understanding and application. Building custom layers confirmed my Module 2 implementations were correct, while using TensorFlow's APIs showed how much boilerplate they eliminate.*
