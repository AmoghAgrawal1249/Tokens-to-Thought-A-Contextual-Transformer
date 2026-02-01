# Week 1: Python Scientific Computing Fundamentals

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

This week focused on building a strong foundation with Python's scientific computing libraries. I worked through exercises in NumPy array manipulation, data visualization with Matplotlib/Pandas, and implemented multivariate gradient descent from scratch.

## Completed Assignments

### 1. NumPy Assignment (`NumPy_Assignment.ipynb`)

Practiced core NumPy operations including:
- **Array initialization** using `reshape`, `zeros`, `ones`, `full`, and `broadcast_to`
- **Random number generation** with the modern `np.random.default_rng()` API for reproducibility
- **Transposition and reshaping** using `swapaxes` and `ravel` for flexible axis manipulation
- **Slicing and indexing** with explicit ranges and `np.ix_` for advanced selection
- **Broadcasting** with in-place operations (`np.add`, `np.multiply` with `out=`)
- **Vectorization benchmark** demonstrating 100x+ speedup over nested loops

Key takeaway: Always prefer vectorized operations over Python loops when working with arrays.

### 2. Matplotlib & Pandas (`Matplotlib_and_pandas.ipynb`)

Completed three visualization exercises using sales data:

| Exercise | Description | Techniques Used |
|----------|-------------|-----------------|
| 1 | Total profit line plot | Dotted lines, circle markers, legend positioning |
| 2 | Multi-product sales comparison | Colormap iteration, grid overlay, multi-column legend |
| 3 | Annual sales pie chart | Percentage + absolute labels, explode effect, equal aspect |

### 3. Multivariate Gradient Descent (`Multivariate_Gradient_Descent.ipynb`)

Implemented gradient descent to find minima of:

$$f(x, y) = x^4 + x^2y^2 - y^2 + y^4 + 6$$

Features of my implementation:
- Clean `objective()` and `gradient()` function separation
- Configurable learning rate, tolerance, and max iterations
- **Backtracking line search** — halves step size when objective increases
- Trajectory logging for visualization
- Returns comprehensive result dictionary with final point, value, iterations, and gradient norm

## Files in This Directory

```
Week1/
├── NumPy_Assignment.ipynb        # NumPy exercises with solutions
├── Matplotlib_and_pandas.ipynb   # Visualization exercises
├── Multivariate_Gradient_Descent.ipynb  # Gradient descent implementation
├── company_sales_data.csv        # Dataset for plotting exercises
└── README.md                     # This file
```

## How to Run

```bash
# Option 1: Google Colab (recommended)
# Upload notebooks to Colab and run cells sequentially

# Option 2: Local Jupyter
pip install numpy matplotlib pandas
jupyter notebook NumPy_Assignment.ipynb
```

## Resources I Found Helpful

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) — great for finding plot types
- [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)

---

*Completed: Week 1 of Tokens-to-Thought course*
