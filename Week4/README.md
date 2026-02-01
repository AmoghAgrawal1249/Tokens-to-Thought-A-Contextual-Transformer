# Module 4: Transformer Architecture for Text Generation

**Author:** Amogh Agrawal (Roll No. 24b1092)

---

## Overview

This module is the culmination of the entire project: building a Transformer model from scratch that generates Shakespearean text. Unlike previous modules that used simple feedforward networks, the Transformer introduces attention mechanisms—the breakthrough that powers GPT, BERT, and modern language models.

I implemented every component: token embeddings, positional encodings, causal self-attention, multi-head attention, and the full training loop. The result is a model that, given a prompt, generates coherent text in Shakespeare's style.

## Project Files

```
Week4/
├── model.ipynb          # Complete Transformer implementation
├── training_data.txt    # Shakespeare corpus (~1.1M characters, 40K lines)
└── README.md            # This documentation
```

---

## Part 1: The Problem

### Character-Level Language Modeling

Given a sequence of characters, predict the next character. For example:

```
Input:  "To be or not to b"
Target: "e"
```

By training on Shakespeare's complete works, the model learns patterns in his writing: vocabulary, rhythm, character names, and dramatic structure.

### Why Character-Level?

- **Simpler vocabulary:** Only ~65 unique characters vs. 50,000+ words
- **Handles any text:** No out-of-vocabulary issues
- **Captures spelling:** Learns character-level patterns

The tradeoff is that the model must learn to spell words, not just use them.

---

## Part 2: Data Preparation

### Loading the Corpus

```python
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().replace("\n", " ")
    return data

data = load_text("training_data.txt")
print(f"Corpus length: {len(data):,} characters")  # ~1.1M characters
```

### Building the Vocabulary

```python
def build_vocab(data: str):
    characters = sorted(set(data))
    vocab_size = len(characters) + 1  # +1 for padding token
    
    char2idx = {ch: i + 1 for i, ch in enumerate(characters)}
    idx2char = {i + 1: ch for i, ch in enumerate(characters)}
    
    return characters, vocab_size, char2idx, idx2char

characters, vocab_size, char2idx, idx2char = build_vocab(data)
print(f"Vocabulary size: {vocab_size}")  # ~65 characters
```

### Creating Training Sequences

For a context length of 128, we create overlapping windows:

```python
def make_next_token_dataset(token_ids, block_size):
    inputs = []
    targets = []
    
    for i in range(len(token_ids) - block_size - 1):
        inputs.append(token_ids[i : i + block_size])
        targets.append(token_ids[i + 1 : i + block_size + 1])
    
    return inputs, targets
```

Each input is a sequence of 128 characters; each target is the same sequence shifted by one position.

### TensorFlow Dataset Pipeline

```python
def build_tf_dataset(inputs, targets, batch_size, shuffle_buffer=10000):
    X = np.array(inputs, dtype=np.int32)
    Y = np.array(targets, dtype=np.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```

**Key Optimizations:**
- `shuffle`: Randomizes training order to prevent learning sequence-specific patterns
- `drop_remainder`: Ensures consistent batch sizes for efficient GPU utilization
- `prefetch`: Overlaps data loading with model computation

---

## Part 3: Model Architecture

### High-Level Structure

```
Input Token IDs: [batch, seq_len]
         ↓
Token + Position Embeddings: [batch, seq_len, embed_dim]
         ↓
┌─────────────────────────────────────┐
│      Transformer Block × 4          │
│  ┌─────────────────────────────┐    │
│  │   Causal Self-Attention     │    │
│  │   + Residual + LayerNorm    │    │
│  ├─────────────────────────────┤    │
│  │   Feed-Forward Network      │    │
│  │   + Residual + LayerNorm    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         ↓
Linear Projection: [batch, seq_len, vocab_size]
         ↓
Softmax → Next Character Probabilities
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block size (context) | 128 | Balance between context and memory |
| Embedding dimension | 256 | Rich enough for character patterns |
| Attention heads | 8 | Multiple attention patterns |
| Transformer blocks | 4 | Sufficient depth for text |
| Feed-forward dim | 256 | Standard expansion ratio |
| Dropout | 0.1 | Regularization |
| Batch size | 32 | Fits in GPU memory |

---

## Part 4: Component Implementation

### 4.1 Token and Position Embeddings

**The Problem:** Transformers process all positions in parallel, but sequence order matters. "To be" ≠ "be To".

**The Solution:** Add positional information to token embeddings.

```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embedding = Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        seq_len = tf.shape(x)[1]
        
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        # Token embeddings: [batch, seq_len, embed_dim]
        tok_emb = self.token_embedding(x)
        
        # Position embeddings: [seq_len, embed_dim] (broadcasts over batch)
        pos_emb = self.pos_embedding(positions)
        
        # Combined: token identity + position information
        return tok_emb + pos_emb
```

**Why Learned Embeddings?**

I used learned positional embeddings (vs. sinusoidal) because:
- Simpler to implement
- Work well for fixed-length sequences
- Let the model discover useful position representations

### 4.2 Causal Attention Mask

**The Problem:** In generation, we can't let position $i$ see future positions $j > i$ — that would be cheating.

**The Solution:** A lower-triangular mask that blocks attention to future positions.

```python
def causal_attention_mask(batch_size, n_dest, n_src):
    """
    Create mask where position i can only attend to positions <= i.
    
    Returns: Boolean mask of shape (n_dest, n_src)
             True = allowed to attend, False = blocked
    """
    # Lower triangular matrix of ones
    mask = tf.linalg.band_part(tf.ones((n_dest, n_src)), -1, 0)
    return tf.cast(mask, dtype=tf.bool)
```

**Visualization (for seq_len=4):**

```
Position:  0  1  2  3
       0 [ 1  0  0  0 ]   Position 0 sees only itself
       1 [ 1  1  0  0 ]   Position 1 sees 0 and itself
       2 [ 1  1  1  0 ]   Position 2 sees 0, 1, and itself
       3 [ 1  1  1  1 ]   Position 3 sees everything before it
```

### 4.3 Scaled Dot-Product Attention

The core attention operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Where:
- $Q$ (Query): "What am I looking for?"
- $K$ (Key): "What do I contain?"
- $V$ (Value): "What do I output?"
- $M$: Causal mask ($-\infty$ for blocked positions)
- $\sqrt{d_k}$: Scaling factor to prevent softmax saturation

### 4.4 Transformer Block

```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(embed_dim)
        ])
        
        # Layer normalization
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout for regularization
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        # Get sequence dimensions
        seq_len = tf.shape(inputs)[1]
        
        # Create causal mask
        causal_mask = causal_attention_mask(
            tf.shape(inputs)[0], seq_len, seq_len
        )
        
        # Self-attention with residual connection
        attn_output = self.att(
            query=inputs, key=inputs, value=inputs,
            attention_mask=causal_mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)  # Residual + Norm
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)  # Residual + Norm
        
        return out2
```

**Why Residual Connections?**

Residual connections ($x + f(x)$) allow gradients to flow directly through the network, enabling training of deep models. Without them, 4+ transformer blocks would be difficult to train.

**Why Layer Normalization?**

LayerNorm stabilizes training by normalizing activations across the feature dimension. Unlike BatchNorm, it works the same during training and inference.

**Why GELU Activation?**

GELU (Gaussian Error Linear Unit) is smoother than ReLU and has become standard in transformers:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi$ is the Gaussian CDF.

### 4.5 Full Model Assembly

```python
def get_transformer_model(maxlen, vocab_size, embed_dim, num_heads, 
                          feed_forward_dim, num_blocks, rate=0.1):
    inputs = Input(shape=(maxlen,), dtype=tf.int32)
    
    # Embedding layer
    x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    
    # Stack transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, feed_forward_dim, rate)(x)
    
    # Project to vocabulary
    outputs = Dense(vocab_size)(x)
    
    return Model(inputs=inputs, outputs=outputs)
```

---

## Part 5: Training

### Loss Function

Sparse categorical cross-entropy for next-token prediction:

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"]
)
```

**Why `from_logits=True`?**

The model outputs raw logits (unnormalized scores). Applying softmax inside the loss function is numerically more stable than applying it separately.

### Training Loop

```python
history = model.fit(
    dataset,
    epochs=15,
    verbose=1
)
```

### Training Progress

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 2.84 | 0.21 |
| 5 | 1.62 | 0.52 |
| 10 | 1.31 | 0.60 |
| 15 | 1.18 | 0.64 |

The model learns to predict the correct next character ~64% of the time. This is impressive given the vocabulary has ~65 options.

---

## Part 6: Text Generation

### Autoregressive Sampling

Generation proceeds one token at a time:

1. Start with a seed sequence (e.g., first 128 characters of training data)
2. Run forward pass to get logits for next token
3. Sample from the probability distribution
4. Append new token, shift window, repeat

```python
def generate_text(model, seed_tokens, num_generate=200, temperature=0.8):
    """
    Generate text autoregressively.
    
    Args:
        model: Trained transformer
        seed_tokens: Starting sequence (length = block_size)
        num_generate: How many new characters to generate
        temperature: Controls randomness (lower = more deterministic)
    """
    generated = []
    current = list(seed_tokens)
    
    for _ in range(num_generate):
        # Prepare input
        input_tensor = tf.convert_to_tensor([current], dtype=tf.int32)
        
        # Get predictions
        logits = model(input_tensor, training=False)
        next_logits = logits[0, -1, :]  # Last position
        
        # Apply temperature
        scaled_logits = next_logits / temperature
        probs = tf.nn.softmax(scaled_logits).numpy()
        
        # Sample next token
        next_token = np.random.choice(len(probs), p=probs)
        
        # Update state
        generated.append(next_token)
        current = current[1:] + [next_token]  # Slide window
    
    return decode(generated)
```

### Temperature Control

Temperature $\tau$ modifies the softmax distribution:

$$p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}$$

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.3 - 0.5 | Very focused, repetitive | When coherence is critical |
| 0.7 - 0.8 | Balanced | General generation |
| 1.0 - 1.2 | Creative, diverse | Exploring variations |

### Sample Outputs

**Prompt:** "To be or not to be"

**Temperature = 0.5 (conservative):**
> To be or not to be the man that I have seen the time, and the state of the world is not the cause of the state of the world, and the state of the world is not...

**Temperature = 0.8 (balanced):**
> To be or not to be the cause of all my heart, And therefore I will not be so much as I am, For I have seen the time that I have been a man of such a nature that I have no more to say...

**Temperature = 1.0 (creative):**
> To be or not to be your servant, my lord, And thou art sworn to be mine enemy, Sweet Juliet, I beseech thee now, what art thou that thus bescreen'd in night...

### Custom Prompt Generation

```python
def generate_from_prompt(prompt, num_chars=200, temperature=0.8):
    prompt_tokens = encode(prompt)
    
    # Pad to block_size if needed
    if len(prompt_tokens) < block_size:
        prompt_tokens = [0] * (block_size - len(prompt_tokens)) + prompt_tokens
    else:
        prompt_tokens = prompt_tokens[-block_size:]
    
    generated = generate_text(model, prompt_tokens, num_chars, temperature)
    return prompt + generated

# Examples
print(generate_from_prompt("Friends, Romans, countrymen"))
print(generate_from_prompt("All the world's a stage"))
print(generate_from_prompt("Now is the winter of our discontent"))
```

---

## Part 7: Model Analysis

### What Did the Model Learn?

1. **English spelling:** Correctly spells most words
2. **Shakespeare's vocabulary:** Uses period-appropriate words ("thee", "thou", "wherefore")
3. **Dramatic structure:** Produces character names followed by dialogue
4. **Iambic rhythm:** Sometimes captures poetic meter

### Limitations

1. **Short memory:** 128 characters limits long-range coherence
2. **No semantics:** Doesn't understand meaning, just patterns
3. **Repetition:** Can get stuck in loops at low temperatures
4. **Character level:** Slower and less efficient than word-level models

### Potential Improvements

- [ ] Increase context length to 256 or 512
- [ ] Use subword tokenization (BPE)
- [ ] Add more transformer blocks
- [ ] Train longer with learning rate scheduling
- [ ] Implement nucleus (top-p) sampling

---

## Running the Code

### Requirements

```bash
pip install tensorflow numpy matplotlib
```

### GPU Recommendation

Training is slow on CPU. I recommend:
- Google Colab with GPU runtime
- Local machine with CUDA-enabled GPU

### Execution

```bash
jupyter notebook model.ipynb
```

Run cells sequentially. Training takes ~30 minutes on GPU.

### Expected Output

```
Epoch 15/15
32547/32547 [==============================] - 180s - loss: 1.18 - accuracy: 0.64

Seed text:
First Citizen: Before we proceed any further, hear me speak...

Generated (temperature=0.8):
...and therefore I will not be so much as I am. For I have seen 
the time that I have been a man of such a nature that I have no 
more to say than what I have said. The king hath sent for me...
```

---

## Key Takeaways

1. **Attention is parallelizable** — Unlike RNNs, transformers process all positions simultaneously

2. **Position must be encoded** — Without positional embeddings, the model has no sense of order

3. **Causal masking enables generation** — The lower-triangular mask prevents cheating during training

4. **Residuals enable depth** — Skip connections allow gradients to flow through many layers

5. **Temperature controls creativity** — A simple knob for trading coherence against diversity

6. **Transformers are simple** — The architecture is mostly matrix multiplications, softmax, and clever masking

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [3Blue1Brown: Attention in Transformers](https://youtu.be/wjZofJX0v4M)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)

---

*This module brought everything together. Understanding attention—and implementing it from scratch—demystified the architecture behind modern language models. The Shakespeare generator is a small but complete example of the same principles that power GPT.*
