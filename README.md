# Differentiable Programming — Autograd Engine, Optimizers & Schedulers From Scratch

*Authors: Matis Braun & Virgile Hermant*

A reverse-mode automatic differentiation engine built from scratch (NumPy only, no PyTorch/TF autograd), used as a foundation to implement and compare gradient-based optimizers, a small neural network, and learning-rate schedulers.

## The `Tensor` Autograd Engine

Each `Tensor` stores a value (`.data`) and an accumulated gradient (`.grad`). Every operation (`+`, `*`, `**`, `relu`, plus standalone `cos_d`, `sigmoid_d`) creates a new `Tensor` node that records its parent nodes (`_prev`) and a local `_backward()` closure encoding that operation's derivative.

`backward()` builds a topological ordering of the computation graph (DFS) and replays it in reverse, applying the chain rule at each node to accumulate gradients (the same principle PyTorch's autograd is built on), here implemented for scalar/elementwise ops only.

## Optimizers

All implemented as subclasses of a common `Optimizer` base class operating directly on `Tensor` gradients :

**SGD**
```text
θ(t+1) = θ(t) − η · ∇L(θ(t))
```

**RMSProp**
```text
θ(t+1) = θ(t) − η / √(E[g²](t) + ε) · ∇L(θ(t))
```

**Adagrad**
```text
θ(t+1) = θ(t) − η / √(G(t) + ε) · ∇L(θ(t))
```

**Adam**
```text
m(t) = β1·m(t) + (1 − β1)·∇L(θ(t))
v(t) = β2·v(t) + (1 − β2)·∇L(θ(t))²
θ(t+1) = θ(t) − η / (√v̂(t) + ε) · m̂(t)
```

**AdamW**
```text
θ(t+1) = θ(t) − η / (√v̂(t) + ε) · m̂(t) − η·λ·θ(t)
```

## Optimizer Evaluation

Each optimizer minimizes a **convex** function `f(x) = (x − 2)²` and a **non-convex** function `f(x) = 3x² − 2x`, starting from the same `x₀`. The descent trajectories are rendered as animated GIFs, alongside loss-vs-step curves comparing all 5 optimizers side by side.

## Neural Network

A minimal one-hidden-unit network :
```text
h = W1·x + b1
ŷ = σ(W2·h + b2)
```
trained with MSE loss `(y − ŷ)²` on both the linear and non-linear datasets, comparing how each optimizer converges.

## Learning-Rate Schedulers

- **`LRScheduler`** base class.
- **`LRSchedulerOnPlateau`** reduces the learning rate by a `factor` once the loss stops improving by at least `threshold` for `patience` consecutive steps, down to a `min_lr` floor (same idea as PyTorch's `ReduceLROnPlateau`).

Re-evaluated on the same convex/non-convex functions and on the neural network training loop, to compare convergence with vs. without scheduling.

## Extensions

- **`SimpleRNN`** : a vanilla recurrent network (`Wx`, `Wh`, `Wy` + biases) built on the same `Tensor` engine, with a full forward to backward to parameter-update loop over sequences.
- **`SimpleNN`** / **`DropoutNN`** : exploratory sketches for a dropout-regularized network

## Results

- Animated GIFs comparing all 5 optimizers on the convex and non-convex toy functions, with and without the LR scheduler.
- Loss curves for the small neural network trained on the linear and non-linear datasets, for every optimizer.
