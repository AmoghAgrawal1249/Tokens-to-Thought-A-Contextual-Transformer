# Tokens-to-Thought: A Contextual Transformer

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Project Overview

This repository documents my journey through a 4-week deep learning course, culminating in the implementation of a character-level Transformer model for Shakespearean text generation. Starting from the fundamentals of NumPy and gradient descent, I progressively built up to implementing neural networks from scratch and finally a complete Transformer architecture.

## Repository Structure

```
├── Week1/                      # NumPy, Matplotlib, Pandas & Gradient Descent
│   ├── NumPy_Assignment.ipynb
│   ├── Matplotlib_and_pandas.ipynb
│   ├── Multivariate_Gradient_Descent.ipynb
│   └── company_sales_data.csv
│
├── Week2/                      # Neural Networks from Scratch
│   └── feedforward_from_scratch.py
│
├── Week3/                      # TensorFlow Fundamentals
│   └── Week3.ipynb
│
├── Week4/                      # Transformer Implementation
│   ├── model.ipynb
│   └── training_data.txt
│
└── Week1_Week2_Report.tex      # LaTeX report for Weeks 1 & 2
```

## Weekly Breakdown

| Week | Focus Area | Key Deliverables |
|------|-----------|------------------|
| 1 | Python Scientific Stack | NumPy operations, Matplotlib visualizations, Gradient descent |
| 2 | Neural Networks from Scratch | Perceptron, AND/XOR gates, Full adder, Ripple-carry adder |
| 3 | TensorFlow Basics | Custom layers, MNIST classifier, Housing price regression |
| 4 | Transformers | Causal attention, Positional embeddings, Shakespeare text generation |

## How to Run

### Prerequisites
```bash
pip install numpy matplotlib pandas tensorflow scikit-learn
```

### Week 1-3
Open the respective Jupyter notebooks in Google Colab or locally:
```bash
jupyter notebook Week1/NumPy_Assignment.ipynb
```

### Week 2 (Standalone Script)
```bash
python Week2/feedforward_from_scratch.py
```

### Week 4 (Transformer)
```bash
jupyter notebook Week4/model.ipynb
```
Note: Training the transformer requires significant compute. GPU acceleration is recommended.

## Key Learnings

1. **Vectorization matters** — The speed difference between loop-based and vectorized NumPy operations is dramatic (100x+)
2. **Single perceptrons have limits** — XOR cannot be solved without hidden layers due to linear separability constraints
3. **Backpropagation is elegant** — The chain rule applied systematically enables learning in deep networks
4. **Attention is powerful** — Self-attention allows models to capture long-range dependencies in sequences

## Acknowledgments

This coursework was part of a structured deep learning program. Resources from 3Blue1Brown, Andrew Ng's courses, and various online tutorials were instrumental in building my understanding.

---

*Last updated: February 2026*
