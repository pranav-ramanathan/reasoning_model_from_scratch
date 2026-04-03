# PyTorch Training Tutorial: From Basics to Chain-of-Thought

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [The Training Loop Explained](#the-training-loop-explained)
3. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
4. [Common Pitfalls](#common-pitfalls)
5. [Your Code Filled In](#your-code-filled-in)

---

## Core Concepts

### What is a Training Loop?

A training loop is the repetitive process that teaches a model. Think of it like studying:
1. **Read** a problem (forward pass)
2. **Check** if you got it right (compute loss)
3. **Understand** where you went wrong (backpropagation)
4. **Practice** the solution again (weight updates)

Repeat thousands of times, and your model gets smarter.

### The Three Pillars

Every PyTorch training involves three things:

1. **Forward Pass**: Feed data through the model, get predictions
2. **Loss Computation**: Measure how wrong the predictions are
3. **Backward Pass + Weight Update**: Fix the mistakes

---

## The Training Loop Explained

### What Happens in Each Step

```python
for epoch in range(NUM_EPOCHS):
    model.train()                    # Put model in training mode
    
    # STEP 1: Move data to GPU
    batch = {k: v.to(device) for k, v in train_data.items()}
    
    # STEP 2: Forward pass
    outputs = model(**batch)         # Feed data through the model
    
    # STEP 3: Extract loss
    loss = outputs.loss              # The model computed this for us
    
    # STEP 4: Backward pass
    optimizer.zero_grad()            # Clear old gradients
    loss.backward()                  # Compute new gradients
    
    # STEP 5: Update weights
    optimizer.step()                 # Move in the direction that reduces loss
    
    losses.append(loss.item())       # Track the loss
```

---

## Step-by-Step Walkthrough

### Step 1: Move Data to Device (GPU)

```python
batch = {k: v.to(device) for k, v in train_data.items()}
```

**What's happening?**
- Your data is probably on CPU (slow for deep learning)
- GPUs are thousands of times faster for matrix operations
- This line copies each tensor in the batch to the GPU

**Why does it matter?**
- Training without a GPU can take 10-100x longer
- The model parameters are already on the GPU (we moved them there earlier)
- Everything needs to be on the same device to compute

**What's in the batch?**
```python
batch = {
    'input_ids': tensor([101, 2054, 2003, ...]),      # Token IDs from text
    'attention_mask': tensor([1, 1, 1, 0, 0, ...]),   # Which tokens are real vs padding
    'labels': tensor([2054, 2003, 1998, ...])          # What the model should predict next
}
```

---

### Step 2: Forward Pass

```python
outputs = model(**batch)
```

**What's the `**` doing?**
- It unpacks the dictionary as keyword arguments
- Equivalent to: `model(input_ids=..., attention_mask=..., labels=...)`

**What happens inside the model?**
1. Tokenized text goes in
2. Embeddings are created (convert token IDs to vectors)
3. Transformer layers process the embeddings
4. Final layer produces logits (raw predictions) for each position
5. Since we passed `labels`, the model automatically computes cross-entropy loss

**What do we get back?**
```python
outputs = {
    'loss': tensor(2.543),           # Average loss across all tokens
    'logits': tensor([...]),         # Raw predictions (shape: batch_size × seq_len × vocab_size)
    'hidden_states': tensor([...])   # Internal representations (if you requested them)
}
```

---

### Step 3: Extract the Loss

```python
loss = outputs.loss
```

**What is loss?**

Loss measures "how wrong was the model?" Lower loss = better predictions.

**The specific loss function (Cross-Entropy):**

For each token position, the model predicts a probability for every word in the vocabulary. We compare this to the correct answer:

```
Model prediction: [0.01, 0.02, 0.95, 0.02, ...]  (probabilities across vocab)
Correct answer:   [0, 0, 1, 0, ...]               (one-hot: token index 2 is correct)

Loss = -log(0.95) ≈ 0.05  (small loss = correct prediction)
vs
Loss = -log(0.01) ≈ 4.6   (large loss = wrong prediction)
```

The model is trained to maximize the probability of correct tokens.

**Why take the log?**
- Numbers get very small very fast (0.95^100 ≈ 0)
- Log makes them tractable
- It's a mathematical trick that works beautifully for probability

---

### Step 4: Backward Pass (Backpropagation)

```python
optimizer.zero_grad()    # Clear old gradients
loss.backward()          # Compute new gradients
```

**What's a gradient?**

A gradient tells you: "If I nudge this weight by a tiny amount, how much does the loss change?"

```
Weight = 0.5
Gradient = -0.3  means:  if I increase the weight, loss decreases
Gradient = 0.2   means:  if I increase the weight, loss increases (so decrease it)
```

**Why `zero_grad()`?**

PyTorch *accumulates* gradients by default. If you don't clear them, old gradients stick around and mess up your updates.

**What does `loss.backward()` do?**

This is the magic. Using calculus (chain rule), it computes:
- How does each model parameter affect the loss?
- It traces backwards through every operation

The math is complex, but the idea is simple: "What changes should I make?"

---

### Step 5: Update Weights

```python
optimizer.step()
```

**What's an optimizer?**

An optimizer uses the gradients to update the weights. Different optimizers update differently:

**Gradient Descent** (simplest):
```python
new_weight = old_weight - learning_rate × gradient
```

If gradient says "decrease the loss by increasing this weight", we increase it.
If gradient says "decrease the loss by decreasing this weight", we decrease it.

**AdamW** (what you're using):
```python
# More complex, but the intuition is similar
# It also uses momentum (remembers past gradients)
# and adaptive learning rates (different weights learn at different speeds)
```

**The learning rate** (you set it to `5e-5`):
- Too high: weights swing wildly, training becomes unstable
- Too low: training is very slow
- This is one of the main hyperparameters you tune

---

## Common Pitfalls

### ❌ Mistake 1: Forgetting `optimizer.zero_grad()`
```python
# WRONG
for epoch in range(10):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    # Gradients accumulate! Next epoch's gradients add to old ones
```

**Why it breaks:** Gradients are cumulative. Without clearing them, each update uses stale information.

### ❌ Mistake 2: Calling `backward()` before `zero_grad()`
```python
# WRONG
loss.backward()              # Compute gradients
loss2.backward()             # Gradient add up
optimizer.step()             # Update using corrupted gradients
optimizer.zero_grad()        # Clear (too late!)
```

**Why it breaks:** The order matters. Clear old gradients first, compute new ones, then update.

### ❌ Mistake 3: Not moving data to device
```python
# WRONG
outputs = model(**train_data)  # train_data is on CPU, model is on GPU
# Error: Expected all tensors to be on the same device
```

### ❌ Mistake 4: Forgetting `model.train()`
```python
# Can work, but risky
outputs = model(**batch)
loss = outputs.loss
```

**Why it matters:** Some layers behave differently in training vs evaluation:
- Dropout (randomly disables neurons) only works in training mode
- Batch normalization uses running statistics in eval mode, but updates them in training

---

## Your Code Filled In

Here's the complete, working code:

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
NUM_EPOCHS = 50

losses = []

for epoch in range(NUM_EPOCHS):
    model.train()

    # STEP 1: Move data to device
    batch = {k: v.to(device) for k, v in train_data.items()}

    # STEP 2: Forward pass — the model computes the loss internally
    # when you pass both input_ids and labels
    outputs = model(**batch)

    # STEP 3: Extract the loss from outputs
    loss = outputs.loss

    # STEP 4: Backward pass (compute gradients)
    optimizer.zero_grad()
    loss.backward()

    # STEP 5: Update weights
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")
```

---

## The Big Picture: Why Does This Work?

### The Intuition

Every time through the loop:
1. We make a prediction on random data
2. We measure how wrong we are
3. We figure out which knobs to turn to reduce that wrongness
4. We turn them slightly

Repeat 10,000 times with different data, and the model learns.

### Why Chain-of-Thought?

Without CoT, a model just learns to predict the final answer. With CoT:
- The model first generates intermediate reasoning steps
- Then generates the final answer
- The loss trains it on *both* (thinking + answer)
- It learns that reasoning before answering is useful

### What Gets Learned?

The model doesn't learn math facts. It learns:
1. **Format**: Where does reasoning go? (Inside `<think>` tags)
2. **Structure**: How are multi-step problems broken down?
3. **Patterns**: "When I see a problem about addition, I usually show the steps"

This is why SFT alone isn't enough (see your notebook's conclusion). The model learns the *structure* of reasoning, but not which reasoning is *correct*. That's where reinforcement learning comes in.

---

## Next Steps

Try these experiments:

1. **Add print statements** to understand what's happening:
   ```python
   print(f"Batch keys: {batch.keys()}")
   print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
   print(f"Loss value: {loss.item()}")
   ```

2. **Watch the loss curve**: Does it decrease smoothly? That's a sign of good training.

3. **Try different learning rates**: What happens at `1e-4` vs `1e-6`?

4. **Look at generated outputs**: After training, does the model produce `<think>` tags?

---

## Questions to Ask Yourself

1. Why do we compute `loss.backward()` instead of just reading the loss value?
2. What would happen if you used `learning_rate=1.0` instead of `5e-5`?
3. Why does the loss decrease over epochs?
4. What would happen if you never called `model.train()`?
5. If you trained for 1000 epochs instead of 50, would the model be better?

The answers deepen your understanding of how models learn.
