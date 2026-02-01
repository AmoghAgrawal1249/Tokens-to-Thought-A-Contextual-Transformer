# Week 4: Transformer for Shakespeare Text Generation

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

The final week brought everything together with the implementation of a Transformer model capable of generating Shakespearean text. This was the most challenging and rewarding part of the course — implementing causal self-attention, positional embeddings, and autoregressive text generation from scratch using TensorFlow.

## The Goal

Train a character-level language model on Shakespeare's works that can generate new text in a similar style. Given a prompt like "To be or not to be", the model continues with plausible Shakespearean prose.

## Model Architecture

```
Input Tokens
     ↓
Token + Position Embedding
     ↓
┌─────────────────────────────┐
│    Transformer Block × 4    │
│  ┌───────────────────────┐  │
│  │ Causal Self-Attention │  │
│  │    + Residual + Norm  │  │
│  ├───────────────────────┤  │
│  │   Feed-Forward (GELU) │  │
│  │    + Residual + Norm  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
     ↓
Linear → Vocab Logits
     ↓
Softmax → Next Token Prediction
```

## Implemented Components

### 1. `causal_attention_mask()`
Creates a lower-triangular boolean mask ensuring position `i` can only attend to positions `≤ i`. This is what makes the model autoregressive — it can't "cheat" by looking at future tokens.

```python
# For sequence length 4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

### 2. `TokenAndPositionEmbedding`
Combines two learned embedding tables:
- **Token embeddings:** Map each character to a 256-dim vector
- **Position embeddings:** Encode where each token sits in the sequence

The sum gives the model both "what" and "where" information.

### 3. `TransformerBlock`
A single transformer layer with:
- Multi-head self-attention (8 heads)
- GELU-activated feed-forward network
- Layer normalization
- Residual connections
- Dropout for regularization

### 4. `build_tf_dataset()`
Prepares training data:
- Creates input/target pairs (predict next character)
- Converts to `tf.data.Dataset`
- Shuffles and batches efficiently
- Uses `prefetch()` for performance

### 5. `generate_text()`
Autoregressive sampling:
- Maintains a sliding window of `block_size` tokens
- Runs forward pass to get next-token probabilities
- Samples from distribution (with temperature control)
- Appends new token and shifts window

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Block size (context length) | 128 |
| Embedding dimension | 256 |
| Number of attention heads | 8 |
| Number of transformer blocks | 4 |
| Feed-forward hidden dim | 256 |
| Dropout rate | 0.1 |
| Batch size | 32 |

## Training

- **Dataset:** 40,000+ lines of Shakespeare plays
- **Training samples:** ~1M character sequences
- **Epochs:** 15
- **Loss:** Sparse categorical cross-entropy

## Text Generation

The model supports temperature-controlled sampling:

| Temperature | Behavior |
|-------------|----------|
| 0.5 | Conservative, repetitive but coherent |
| 0.8 | Balanced creativity and coherence |
| 1.0 | More varied, occasionally surprising |

### Example Output (after training)

**Prompt:** "To be or not to be"

**Generated (temp=0.7):**
```
To be or not to be the cause of all my heart,
And therefore I will not be so much as I am,
For I have seen the time that I have been...
```

## Files in This Directory

```
Week4/
├── model.ipynb         # Complete transformer implementation
├── training_data.txt   # Shakespeare corpus (~40k lines)
└── README.md           # This file
```

## How to Run

```bash
# Install dependencies
pip install tensorflow numpy matplotlib

# Run notebook (GPU recommended)
jupyter notebook model.ipynb
```

**Note:** Training takes significant time on CPU. Google Colab with GPU runtime is recommended.

## Key Learnings

1. **Attention is all you need** — Self-attention allows every position to directly interact with every other position (subject to causal mask)

2. **Positional encoding is crucial** — Without it, the model has no sense of token order

3. **Temperature controls creativity** — Lower values make output more deterministic, higher values more random

4. **Residual connections enable depth** — They allow gradients to flow through many layers without vanishing

5. **The transformer is surprisingly simple** — It's mostly matrix multiplications, softmax, and clever masking

## Resources I Found Helpful

- [3Blue1Brown: Attention in Transformers](https://youtu.be/wjZofJX0v4M)
- [3Blue1Brown: How LLMs Work](https://youtu.be/eMlx5fFNoYc)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy's makemore](https://github.com/karpathy/makemore)

---

*Completed: Week 4 of Tokens-to-Thought course*

**This marks the completion of the full 4-week journey from NumPy basics to building a working Transformer!**
