import numpy as np


def step_activation(values: np.ndarray) -> np.ndarray:
    return (values >= 0).astype(int)


class Perceptron:
    def __init__(self, n_features: int, learning_rate: float = 0.1, epochs: int = 50, include_bias: bool = True):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.include_bias = include_bias
        self.weights = None
        self.bias = None
        self.initialize()

    def initialize(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0 if self.include_bias else None

    def net_input(self, inputs: np.ndarray) -> np.ndarray:
        linear = inputs @ self.weights
        if self.include_bias:
            linear = linear + self.bias
        return linear

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float64)
        linear = self.net_input(inputs)
        return step_activation(linear)

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        inputs = np.asarray(inputs, dtype=np.float64)
        targets = np.asarray(targets, dtype=int)
        self.initialize()
        for _ in range(self.epochs):
            errors = 0
            for sample, target in zip(inputs, targets):
                linear_output = float(np.dot(sample, self.weights) + (self.bias if self.include_bias else 0.0))
                prediction = 1 if linear_output >= 0 else 0
                error = target - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * sample
                    if self.include_bias:
                        self.bias += self.learning_rate * error
                    errors += 1
            if errors == 0:
                break


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def sigmoid_derivative(activated: np.ndarray) -> np.ndarray:
    return activated * (1.0 - activated)


def tanh_derivative(activated: np.ndarray) -> np.ndarray:
    return 1.0 - activated**2


def initialize_parameters(input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator) -> dict:
    limit_hidden = np.sqrt(6 / (input_dim + hidden_dim))
    limit_output = np.sqrt(6 / (hidden_dim + output_dim))
    return {
        "W1": rng.uniform(-limit_hidden, limit_hidden, size=(input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim)),
        "W2": rng.uniform(-limit_output, limit_output, size=(hidden_dim, output_dim)),
        "b2": np.zeros((1, output_dim)),
    }


def forward_propagation(inputs: np.ndarray, parameters: dict) -> dict:
    z1 = inputs @ parameters["W1"] + parameters["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    a2 = sigmoid(z2)
    return {"A0": inputs, "Z1": z1, "A1": a1, "Z2": z2, "A2": a2}


def compute_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    loss = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return float(np.mean(loss))


def backward_propagation(targets: np.ndarray, cache: dict, parameters: dict) -> dict:
    m = targets.shape[0]
    dz2 = cache["A2"] - targets
    dW2 = cache["A1"].T @ dz2 / m
    db2 = np.mean(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ parameters["W2"].T) * tanh_derivative(cache["A1"])
    dW1 = cache["A0"].T @ dz1 / m
    db1 = np.mean(dz1, axis=0, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(parameters: dict, gradients: dict, learning_rate: float) -> dict:
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters


class TwoLayerNetwork:
    def __init__(self, hidden_units: int, learning_rate: float = 0.5, epochs: int = 8000, seed: int | None = None):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.parameters = None
        self.loss_history = []

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        inputs = np.asarray(inputs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        if self.parameters is None:
            self.parameters = initialize_parameters(inputs.shape[1], self.hidden_units, targets.shape[1], self.rng)

        self.loss_history = []
        for _ in range(self.epochs):
            cache = forward_propagation(inputs, self.parameters)
            loss = compute_loss(cache["A2"], targets)
            self.loss_history.append(loss)
            gradients = backward_propagation(targets, cache, self.parameters)
            self.parameters = update_parameters(self.parameters, gradients, self.learning_rate)

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float64)
        cache = forward_propagation(inputs, self.parameters)
        return cache["A2"]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(inputs)
        return (probabilities >= 0.5).astype(int)


def run_and_gate_demo() -> None:
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    targets = np.array([0, 0, 0, 1], dtype=int)
    perceptron = Perceptron(n_features=2, learning_rate=0.2, epochs=20)
    perceptron.fit(data, targets)
    predictions = perceptron.predict(data)
    print("AND gate truth table using a single perceptron:")
    for sample, prediction in zip(data, predictions):
        print(f"Input {tuple(sample.astype(int))} -> {prediction}")


def run_single_perceptron_xor_attempt() -> None:
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    targets = np.array([0, 1, 1, 0], dtype=int)
    perceptron = Perceptron(n_features=2, learning_rate=0.2, epochs=50)
    perceptron.fit(data, targets)
    predictions = perceptron.predict(data)
    mismatches = predictions != targets
    print("\nAttempting XOR with a single perceptron:")
    for sample, prediction, target in zip(data, predictions, targets):
        mark = "✗" if prediction != target else "✓"
        print(f"Input {tuple(sample.astype(int))} -> predicted {prediction}, target {target} {mark}")
    if mismatches.any():
        print("Result: A single perceptron fails to model the XOR gate due to non-linear separability.")


def run_two_layer_xor_demo() -> TwoLayerNetwork:
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    targets = np.array([0, 1, 1, 0], dtype=int)
    network = TwoLayerNetwork(hidden_units=4, learning_rate=0.6, epochs=5000, seed=42)
    network.fit(data, targets)
    predictions = network.predict(data)
    print("\nXOR gate using a two-layer neural network:")
    for sample, prediction in zip(data, predictions):
        print(f"Input {tuple(sample.astype(int))} -> {prediction.item()}")
    return network


def run_full_adder_demo() -> TwoLayerNetwork:
    data = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 1],
        ],
        dtype=int,
    )
    network = TwoLayerNetwork(hidden_units=6, learning_rate=0.8, epochs=8000, seed=7)
    network.fit(data, targets)
    predictions = network.predict(data)
    print("\nFull adder (Sum, Carry) using a two-layer neural network:")
    for sample, prediction in zip(data, predictions):
        print(f"Input {tuple(sample.astype(int))} -> Sum {prediction[0]}, Carry {prediction[1]}")
    return network


def bits_to_int(bits: np.ndarray) -> int:
    return int("".join(map(str, bits[::-1].astype(int))), 2)


def run_ripple_carry_demo(full_adder_network: TwoLayerNetwork) -> None:
    def ripple_add(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        max_len = max(lhs.size, rhs.size)
        lhs_bits = np.pad(lhs.astype(int), (0, max_len - lhs.size))
        rhs_bits = np.pad(rhs.astype(int), (0, max_len - rhs.size))
        carry = 0
        result_bits = []
        for bit_lhs, bit_rhs in zip(lhs_bits, rhs_bits):
            inputs = np.array([[bit_lhs, bit_rhs, carry]], dtype=np.float64)
            sum_bit, carry_bit = full_adder_network.predict(inputs)[0]
            result_bits.append(sum_bit)
            carry = carry_bit
        result_bits.append(carry)
        return np.array(result_bits, dtype=int)

    lhs = np.array([1, 0, 1, 1], dtype=int)  # binary 1101 -> 13
    rhs = np.array([1, 1, 0, 1], dtype=int)  # binary 1011 -> 11
    total_bits = ripple_add(lhs, rhs)
    print("\nRipple carry addition using the learned full adder network:")
    print(f"Operands (LSB→MSB): {lhs} + {rhs}")
    print(f"Result bits (LSB→MSB): {total_bits}")
    print(f"Decimal check: {bits_to_int(lhs)} + {bits_to_int(rhs)} = {bits_to_int(total_bits)}")


if __name__ == "__main__":
    run_and_gate_demo()
    run_single_perceptron_xor_attempt()
    _ = run_two_layer_xor_demo()
    full_adder_network = run_full_adder_demo()
    run_ripple_carry_demo(full_adder_network)

