# Tokens-to-Thought: A Contextual Transformer

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## About This Project

This repository is a self-directed deep learning project where I built a character-level Transformer model from the ground up. Rather than jumping straight into high-level frameworks, I took a methodical approach—starting with NumPy fundamentals, implementing neural networks from scratch, learning TensorFlow's internals, and finally constructing a complete Transformer architecture capable of generating Shakespearean text.

The project is organized into four progressive modules, each building on concepts from the previous one. By the end, I had a working language model that could generate coherent text in Shakespeare's style.

## Project Structure

```
Tokens-to-Thought-A-Contextual-Transformer/
│
├── Week1/                              # Foundation: Scientific Computing
│   ├── NumPy_Assignment.ipynb          # Array operations, broadcasting, vectorization
│   ├── Matplotlib_and_pandas.ipynb     # Data visualization techniques
│   ├── Multivariate_Gradient_Descent.ipynb  # Optimization from scratch
│   ├── company_sales_data.csv          # Sample dataset for visualization
│   └── README.md                       # Detailed documentation
│
├── Week2/                              # Core: Neural Networks from Scratch
│   ├── feedforward_from_scratch.py     # Complete NN implementation (NumPy only)
│   └── README.md                       # Architecture details & results
│
├── Week3/                              # Framework: TensorFlow Deep Dive
│   ├── Week3.ipynb                     # Custom layers, MNIST, regression
│   └── README.md                       # Implementation notes
│
├── Week4/                              # Capstone: Transformer Implementation
│   ├── model.ipynb                     # Full transformer with attention
│   ├── training_data.txt               # Shakespeare corpus (~1.1M characters)
│   └── README.md                       # Architecture & generation details
│
├── Final_Report.tex                    # Comprehensive LaTeX report
└── README.md                           # This file
```

## Module Overview

### Module 1: Scientific Computing Foundation
Established proficiency with Python's numerical computing stack. Implemented vectorized operations achieving 100x+ speedups over naive loops. Built a multivariate gradient descent optimizer with adaptive learning rates.

### Module 2: Neural Networks from First Principles
Implemented feedforward networks using only NumPy—no frameworks allowed. Built perceptrons, solved the XOR problem with hidden layers, and created a neural network that functions as a ripple-carry adder.

### Module 3: TensorFlow Internals
Deconstructed TensorFlow by implementing custom layer classes. Applied knowledge to MNIST digit classification (97%+ accuracy) and housing price regression, comparing linear vs. non-linear models.

### Module 4: Transformer Architecture
Built a complete Transformer with causal self-attention, positional embeddings, and multi-head attention. Trained on Shakespeare's works to generate stylistically consistent prose.

## Technical Highlights

| Component | Implementation Details |
|-----------|----------------------|
| **Vectorization** | Achieved 120x speedup using NumPy broadcasting over nested loops |
| **Backpropagation** | Derived and implemented gradients for tanh and sigmoid activations |
| **Custom Layers** | Built `Dense`, `Flatten`, `ReLU`, `Softmax` layers from scratch in TensorFlow |
| **Attention Mechanism** | Implemented scaled dot-product attention with causal masking |
| **Text Generation** | Autoregressive sampling with temperature-controlled diversity |

## Results Summary

| Task | Model | Performance |
|------|-------|-------------|
| XOR Gate | 2-layer NN (4 hidden) | 100% accuracy |
| 4-bit Addition | Ripple-carry adder | Correct on all inputs |
| MNIST Classification | Custom layers | 97.2% test accuracy |
| Housing Regression | Feedforward NN | 0.26 MSE (vs 0.52 linear) |
| Text Generation | Transformer | Coherent Shakespearean output |

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib pandas tensorflow scikit-learn
```

### Run Neural Network Demos (Week 2)
```bash
cd Week2
python feedforward_from_scratch.py
```

### Train Transformer (Week 4)
```bash
jupyter notebook Week4/model.ipynb
# Recommend using Google Colab with GPU for faster training
```

## Sample Generated Text

**Prompt:** "To be or not to be"

**Output (temperature=0.7):**
> To be or not to be the cause of all my heart, And therefore I will not be so much as I am, For I have seen the time that I have been a man of such a nature that I have no more to say than what I have said...

## Key Learnings

1. **Vectorization is essential** — NumPy's array operations are not just convenient, they're necessary for practical neural network training

2. **Linear separability matters** — Understanding why XOR fails with a single perceptron clarifies the need for deep architectures

3. **Frameworks abstract complexity** — Implementing custom TensorFlow layers revealed what `Dense()` actually computes

4. **Attention enables parallelism** — Unlike RNNs, Transformers process all positions simultaneously, enabled by the attention mechanism

## Repository Statistics

- **Total Lines of Code:** ~2,500
- **Languages:** Python (100%)
- **Frameworks:** NumPy, TensorFlow, Matplotlib, Pandas
- **Training Data:** 1.1M characters of Shakespeare

## Future Improvements

- [ ] Implement word-level tokenization (BPE or WordPiece)
- [ ] Add sinusoidal positional encodings as alternative
- [ ] Experiment with different attention patterns (sliding window, sparse)
- [ ] Scale up model size and training data
- [ ] Implement beam search for generation

## License

This project is for educational purposes. Feel free to use the code as reference for your own learning.

---

*Built with curiosity and lots of gradient descent.*
