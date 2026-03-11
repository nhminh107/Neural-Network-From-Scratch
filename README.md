# MNIST MLP From Scratch

This project implements a **fully connected neural network from scratch** to classify handwritten digits from the MNIST dataset. The goal of this project is to understand how deep learning frameworks work internally by manually implementing forward propagation, backpropagation, gradient descent, cross entropy loss, softmax classifier, and a data loading pipeline. The model uses **PyTorch tensors only** without using `torch.nn`, `torch.optim`, or `autograd`. All gradients are computed manually.

---
<img width="444" height="348" alt="image" src="https://github.com/user-attachments/assets/6675824f-694d-4ab7-abc9-60f02d49211a" />

# Project Structure

mnist-mlp-from-scratch
│
├── Data.py        # Custom DataLoader and dataset utilities
├── Model.py       # Neural network implementation
├── Pipeline.py    # Training and evaluation pipeline
├── main.py        # Entry point
└── README.md

---

# Dataset

This project uses the **MNIST dataset**, a classic dataset of handwritten digits.

Dataset properties:

Training samples: 60,000
Test samples: 10,000
Image size: 28 × 28
Classes: 10 digits (0–9)

Each image is flattened into a vector before being fed into the network.

28 × 28 → 784

---

# Model Architecture

The neural network is a **3-layer fully connected network**.

Input Layer: 784
Hidden Layer 1: 128
Hidden Layer 2: 64
Output Layer: 10

Architecture:

784 → 128 → 64 → 10

---

# Forward Propagation

Forward propagation computes predictions using matrix multiplication.

Layer 1:

Z₁ = XW₁ + b₁
A₁ = ReLU(Z₁)

Layer 2:

Z₂ = A₁W₂ + b₂
A₂ = ReLU(Z₂)

Output Layer:

Z₃ = A₂W₃ + b₃
ŷ = Softmax(Z₃)

Where:

X = input batch
W = weights
b = biases

---

# Activation Function

The model uses **ReLU activation** to introduce non-linearity.

ReLU(x) = max(0, x)

---

# Softmax Function

Softmax converts raw scores into probabilities.

Softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)

Example output:

[0.01, 0.02, 0.90, 0.01, ...]

The output represents a probability distribution over 10 classes.

---

# Loss Function

We use **Cross Entropy Loss** for multi-class classification.

L = - Σ y * log(ŷ)

Where:

y = true label (one-hot vector)
ŷ = predicted probability

Batch loss:

Loss = (1/m) Σ Lᵢ

Where m is the batch size.

---

# Backpropagation

Backpropagation computes gradients using the **chain rule**.

Output Layer:

dZ₃ = ŷ − y

dW₃ = (A₂ᵀ · dZ₃) / m

db₃ = Σ dZ₃ / m

Hidden Layer 2:

dA₂ = dZ₃ W₃ᵀ

dZ₂ = dA₂ ⊙ ReLU'(Z₂)

dW₂ = (A₁ᵀ · dZ₂) / m

db₂ = Σ dZ₂ / m

Hidden Layer 1:

dA₁ = dZ₂ W₂ᵀ

dZ₁ = dA₁ ⊙ ReLU'(Z₁)

dW₁ = (Xᵀ · dZ₁) / m

db₁ = Σ dZ₁ / m

Derivative of ReLU:

ReLU'(x) =
1 if x > 0
0 otherwise

---

# Weight Update

Weights are updated using **Stochastic Gradient Descent (SGD)**.

W = W − η ∇W

b = b − η ∇b

Where η is the learning rate.

---

# Training Pipeline

Training process:

Load batch
↓
Flatten image (28×28 → 784)
↓
Forward propagation
↓
Compute loss
↓
Backward propagation
↓
Update weights
↓
Repeat for next batch

Training runs for multiple epochs until the loss decreases.

---

# Prediction

Prediction uses the class with highest probability.

argmax(softmax output)

Example:

[0.01, 0.02, 0.90, ...] → predicted digit = 2

---

# Accuracy

Model performance is measured using accuracy.

Accuracy = correct predictions / total samples

A typical MLP trained on MNIST achieves around:

90% – 95% test accuracy.

---

# GPU Support

The model can run on GPU using PyTorch CUDA support.

Example:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

All tensors and model parameters must be moved to the same device.

---

# How to Run

Install dependencies:

pip install torch torchvision

Run the program:

python main.py

Example training output:

Epoch 1 Loss: 2.12
Epoch 2 Loss: 1.35
Epoch 3 Loss: 0.75
Epoch 4 Loss: 0.45
Epoch 5 Loss: 0.32

Test Accuracy: 0.93

---

# Educational Purpose

This project is designed for educational purposes to demonstrate how deep learning frameworks work internally. By implementing everything manually, we gain a deeper understanding of gradient computation, neural network training, backpropagation, and optimization.

---

# Possible Improvements

Future improvements may include:

Dropout
Batch Normalization
Convolutional Neural Networks (CNN)
Adam optimizer
Learning rate scheduling
Training visualization

---

# References

MNIST Dataset
Deep Learning - Ian Goodfellow
Stanford CS231n
