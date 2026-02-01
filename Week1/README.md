# Module 1: Scientific Computing Foundation

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

Before diving into neural networks, I needed a solid foundation in Python's scientific computing ecosystem. This module covers three essential areas: NumPy for numerical operations, Matplotlib/Pandas for visualization, and gradient-based optimization. The goal was not just to use these tools, but to understand them deeply enough that implementing neural networks from scratch would feel natural.

## Project Files

```
Week1/
├── NumPy_Assignment.ipynb              # Comprehensive NumPy exercises
├── Matplotlib_and_pandas.ipynb         # Data visualization project
├── Multivariate_Gradient_Descent.ipynb # Optimization algorithm implementation
├── company_sales_data.csv              # Dataset for visualization exercises
└── README.md                           # This documentation
```

---

## Part 1: NumPy Mastery

### Why NumPy Matters

NumPy is the backbone of scientific Python. Every deep learning framework—TensorFlow, PyTorch, JAX—builds on NumPy's array paradigm. Understanding NumPy means understanding how neural networks actually compute.

### Topics Covered

#### 1.1 Array Creation and Initialization

I explored multiple ways to create arrays, each with different use cases:

```python
# Method 1: Direct creation with reshape
arr = np.array([1, 2, 4, 7, 13, 21], dtype=np.int64).reshape(2, 3)

# Method 2: Using the modern random generator (reproducible)
rng = np.random.default_rng(seed=42)
random_matrix = rng.random((n_rows, n_columns))

# Method 3: Constant arrays via broadcasting
zeros = np.broadcast_to(np.array(0), (4, 5, 2)).copy()
ones = np.full((4, 5, 2), fill_value=1, dtype=np.int64)

# Method 4: Special matrices
identity = np.eye(5)
diagonal = np.diag([1, 2, 3, 4])
```

**Key Insight:** `np.broadcast_to` creates a view without allocating memory, but the result is read-only. Adding `.copy()` creates a writable array.

#### 1.2 Indexing and Slicing

NumPy's advanced indexing is powerful but has subtle behaviors:

```python
# Basic slicing (creates a view, not a copy!)
arr = np.array([4, 1, 5, 6, 11])
middle = arr[1:4]  # [1, 5, 6]

# Fancy indexing with arrays
indices = np.array([0, 2, 4])
selected = arr[indices]  # [4, 5, 11]

# Multi-dimensional slicing
matrix = np.array([[4, 5, 2], [3, 7, 9], [1, 4, 5], [6, 6, 1]])
submatrix = matrix[np.ix_([0, 1, 2], [1, 2])]  # Rows 0-2, columns 1-2
```

**Key Insight:** Slicing creates views (shared memory), while fancy indexing creates copies. This matters for memory efficiency and avoiding bugs.

#### 1.3 Broadcasting

Broadcasting is NumPy's mechanism for operating on arrays of different shapes:

```python
# Scalar broadcast
arr = np.array([1, 2, 3, 4])
np.add(arr, 10, out=arr)  # In-place addition: [11, 12, 13, 14]

# Row-wise multiplication via broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
multipliers = np.array([[4], [5]])  # Shape (2, 1)
result = np.multiply(matrix, multipliers, out=matrix)
# Result: [[4, 8, 12], [20, 25, 30]]
```

**Broadcasting Rules:**
1. Arrays are compared element-wise from the trailing dimensions
2. Dimensions are compatible if they're equal or one of them is 1
3. The result shape is the maximum along each dimension

#### 1.4 Transposition and Reshaping

Understanding array layout is crucial for matrix operations:

```python
# Transpose methods
y = np.array([[1, 2, 3], [4, 5, 6]])
y_T = y.T                    # Shorthand
y_T = np.transpose(y)        # Explicit function
y_T = np.swapaxes(y, 0, 1)   # General axis swapping

# Flattening
flat = y.ravel()             # View when possible
flat = y.flatten()           # Always creates copy
column = y.ravel().reshape(-1, 1)  # Column vector
```

#### 1.5 Vectorization Benchmark

I measured the performance difference between loop-based and vectorized operations:

```python
import time

arr = np.random.rand(1000, 1000)

# Non-vectorized (nested loops)
start = time.time()
result_loop = [[3 * val for val in row] for row in arr]
loop_time = time.time() - start

# Vectorized
start = time.time()
result_vec = arr * 3
vec_time = time.time() - start

print(f"Loop: {loop_time*1000:.2f} ms")
print(f"Vectorized: {vec_time*1000:.2f} ms")
print(f"Speedup: {loop_time/vec_time:.1f}x")
```

**Results:**
| Approach | Time (1000×1000) | Relative |
|----------|------------------|----------|
| Nested loops | 94.76 ms | 1.0x |
| Vectorized | 0.77 ms | 123x faster |

This 123x speedup becomes critical when training neural networks on millions of samples over hundreds of epochs.

---

## Part 2: Data Visualization

### Project: Sales Data Analysis

Using a company's monthly sales data, I created three visualizations demonstrating different aspects of Matplotlib:

#### Exercise 1: Profit Trend Line Chart

**Requirements:**
- Dotted red line with circle markers
- Legend in lower right
- Proper axis labels

```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    df['month_number'],
    df['total_profit'],
    linestyle=':',           # Dotted line
    linewidth=3,
    color='crimson',
    marker='o',
    markerfacecolor='red',
    markeredgecolor='black',
    markersize=8,
    label='Total Profit'
)
ax.set_xlabel('Month Number')
ax.set_ylabel('Profit in dollar')
ax.set_title('Monthly Total Profit Trend')
ax.legend(loc='lower right')
```

#### Exercise 2: Multi-Product Sales Comparison

**Challenge:** Plot each product's monthly sales with distinct colors and markers.

```python
product_columns = [col for col in df.columns 
                   if col not in {'total_profit', 'month_number', 'total_units'}]

fig, ax = plt.subplots(figsize=(12, 6))
colormap = plt.cm.get_cmap('tab10', len(product_columns))

for idx, column in enumerate(product_columns):
    ax.plot(
        df['month_number'],
        df[column],
        label=column.replace('_sales', '').title(),
        linestyle='-',
        linewidth=2,
        marker='s',
        markersize=5,
        color=colormap(idx)
    )

ax.set_xlabel('Month Number')
ax.set_ylabel('Units Sold')
ax.set_title('Monthly Sales by Product')
ax.legend(loc='upper left', ncol=2)
ax.grid(True, linestyle='--', alpha=0.4)
```

#### Exercise 3: Annual Sales Pie Chart

**Challenge:** Show each product's contribution to total annual sales with percentages.

```python
# Aggregate annual totals
annual_totals = df.loc[:, product_columns].sum().sort_values(ascending=False)

# Custom formatter showing both percentage and absolute values
def autopct_format(values):
    total = values.sum()
    def formatter(pct):
        absolute = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({absolute:,})"
    return formatter

fig, ax = plt.subplots(figsize=(10, 10))
explode = np.linspace(0.02, 0.08, len(annual_totals))

ax.pie(
    annual_totals,
    labels=annual_totals.index,
    autopct=autopct_format(annual_totals),
    startangle=90,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 10, 'color': 'navy'}
)
ax.set_title('Annual Sales Distribution by Product')
ax.axis('equal')
```

---

## Part 3: Multivariate Gradient Descent

### The Optimization Problem

I implemented gradient descent to find minima of the function:

$$f(x, y) = x^4 + x^2 y^2 - y^2 + y^4 + 6$$

This function has multiple local minima, making it an interesting test case for optimization.

### Mathematical Foundation

The gradient (vector of partial derivatives) is:

$$\nabla f(x, y) = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix} = \begin{pmatrix} 4x^3 + 2xy^2 \\ 4y^3 - 2y + 2x^2 y \end{pmatrix}$$

Gradient descent iteratively updates the position:

$$\mathbf{p}_{t+1} = \mathbf{p}_t - \alpha \nabla f(\mathbf{p}_t)$$

where $\alpha$ is the learning rate.

### Implementation Details

```python
def objective(point: np.ndarray) -> float:
    """Compute f(x, y)"""
    x, y = point
    return x**4 + (x**2) * (y**2) - y**2 + y**4 + 6

def gradient(point: np.ndarray) -> np.ndarray:
    """Compute analytical gradient"""
    x, y = point
    dx = 4 * x**3 + 2 * x * (y**2)
    dy = 4 * y**3 - 2 * y + 2 * y * (x**2)
    return np.array([dx, dy], dtype=np.float64)

def gradient_descent(start, learning_rate=0.1, tolerance=1e-6, max_iter=10000):
    """
    Gradient descent with backtracking line search.
    
    If a step increases the objective, the learning rate is halved
    until improvement is found (or minimum step size reached).
    """
    point = np.array(start, dtype=np.float64)
    trajectory = [point.copy()]
    step = learning_rate
    
    for iteration in range(1, max_iter + 1):
        grad = gradient(point)
        grad_norm = np.linalg.norm(grad)
        
        # Convergence check
        if grad_norm <= tolerance:
            break
        
        # Propose new point
        candidate = point - step * grad
        
        # Backtracking: if objective increased, reduce step size
        while objective(candidate) > objective(point) and step > 1e-10:
            step *= 0.5
            candidate = point - step * grad
        
        point = candidate
        trajectory.append(point.copy())
    
    return {
        "point": point,
        "value": objective(point),
        "iterations": iteration,
        "gradient_norm": grad_norm,
        "trajectory": np.vstack(trajectory)
    }
```

### Results from Different Starting Points

| Start Point | Converged To | Iterations | Final Value |
|-------------|--------------|------------|-------------|
| (1.0, 1.0) | (0, 0.707) | 847 | 5.75 |
| (-2.0, -2.0) | (0, -0.707) | 1203 | 5.75 |
| (50.0, 50.0) | (0, 0.707) | 2841 | 5.75 |

The function has two symmetric global minima at $(0, \pm\frac{1}{\sqrt{2}})$ with value $f = 5.75$.

### Key Features of My Implementation

1. **Backtracking Line Search:** Automatically reduces step size when stuck
2. **Trajectory Logging:** Records full path for visualization
3. **Tolerance-Based Stopping:** Halts when gradient is sufficiently small
4. **Robust to Large Starts:** Works even from (50, 50) due to adaptive stepping

---

## Running the Code

### Requirements
```bash
pip install numpy matplotlib pandas jupyter
```

### Execute Notebooks
```bash
# Option 1: Jupyter Lab (recommended)
jupyter lab

# Option 2: Classic Jupyter
jupyter notebook NumPy_Assignment.ipynb
```

### Expected Output

**NumPy Benchmark:**
```
Time taken in non-vectorized approach: 94.76 ms
Time taken in vectorized approach: 0.77 ms
```

**Gradient Descent:**
```
Starting point: [1.0, 1.0]
Converged to: [2.77e-08, 0.7071]
Final value: 5.75
Iterations: 847
```

---

## Key Takeaways

1. **NumPy views vs copies:** Understanding when operations share memory prevents subtle bugs

2. **Broadcasting rules:** Once internalized, broadcasting makes code cleaner and faster

3. **Vectorization is non-negotiable:** 100x+ speedups are typical; loops should be avoided in hot paths

4. **Adaptive optimization:** Simple tricks like backtracking prevent divergence without complex algorithms

5. **Visualization matters:** Good plots reveal patterns that raw numbers hide

---

## References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Gradient Descent Visualization](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown

---

*This module established the computational foundation for everything that follows.*
