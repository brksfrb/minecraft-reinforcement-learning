# Minecraft Bot Reinforcement Learning

## Overview
This project implements a reinforcement learning framework where a virtual bot learns to navigate toward a target in a simplified Minecraft‑like environment.  
The environment simulates positions, rotations, and actions (move forward, rotate left/right).  
Custom neural networks are used to train the bot with rewards based on distance and angle improvements.

## Project Structure
- **data_generator.py**  
  Generates simple datasets for testing neural networks.

- **virtual_environment.py**  
  Defines the `Bot`, `Target`, and `MinecraftVENV` classes.  
  Provides functions for randomizing positions, calculating distances, and simulating actions.

- **model.py**  
  Basic neural network with tanh activation and reinforcement learning updates.

- **model_v2.py**  
  Experimental neural network using ReLU and tanh, trained on XOR‑like datasets.

- **model_v3.py**  
  NumPy‑based neural network with ReLU hidden layers and sigmoid output.  
  Includes training loop and prediction functions.

- **run.py**  
  Main training script.  
  Runs episodes where the bot interacts with the environment, receives rewards, and updates the neural network.

- **test_model.py**  
  Loads a saved model and tests it in the environment with fixed positions.

## Features
- Custom reinforcement learning loop with epsilon‑greedy exploration.
- Multiple neural network implementations for experimentation.
- Environment simulating bot movement, rotation, and target tracking.
- Reward shaping based on distance, angle alignment, and reaching the target.

## Usage
1. **Train the model:**
   ```bash
   python run.py