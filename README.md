# babyGPT

## Overview
This repository contains a minimal implementation of a Generative Pre-trained Transformer (GPT) model using PyTorch. The model is designed to learn from a given text input, generate text, and evaluate its performance through training and testing phases. The code is structured for clarity and educational purposes, making it suitable for those interested in understanding the fundamentals of transformer architectures.

## Features
- Character-level Tokenization: The model processes input text at the character level, allowing for a flexible vocabulary.
- Masked Multi-Head Attention: Implements the core attention mechanism that allows the model to focus on different parts of the input sequence.
- Feedforward Neural Network: A simple feedforward network that processes the output from the attention layers.
- Text Generation: Capable of generating text based on learned patterns after training.

## Data Preparation
The model expects an input text file located at data/input.txt. Ensure that this file exists and contains the text data you wish to train on.

## Training the Model
The script contains a basic training loop that runs for a specified number of steps. You can modify parameters such as nheads, nblocks, lr, token_dim, batch_size, and train_steps to experiment with different configurations. To start training, simply run the script: ```python gpt.py```

## Generating Text
After training, the model can generate text by calling the generate method. The script includes an example of generating 500 tokens of text.
Code Structure

## Key Components

- Data Loading and Preprocessing:
    - Loads input data from a specified file.
    - Splits data into training and testing sets.
    - Defines encoding and decoding functions for character mapping.
- Model Architecture:
    - Head: Implements single attention head functionality.
    - MultiHeadMaskedAttention: Combines multiple heads into a single attention mechanism.
    - Block: Represents a transformer block containing multi-head attention and feedforward layers.
    - babyGPT: The main GPT model that combines embeddings, transformer blocks, and linear output layers.
- Training Loop:
    - Randomly samples batches from training data.
    - Computes loss using cross-entropy.
    - Updates model weights using AdamW optimizer.
      
### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
This implementation was primarily inspired by the one and only Andrej Karpathy. Huge shout out and respect to him!
