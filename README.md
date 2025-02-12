# Curiosity-driven Exploration by Self-supervised Prediction

This repository contains the implementation of the Intrinsic Curiosity Module (ICM) combined with the Proximal Policy Optimization (PPO) algorithm. The project explores how curiosity-driven intrinsic rewards can enhance learning in environments with sparse rewards.

## Overview

- **PPO Algorithm**: An online policy gradient algorithm optimized for stability.
- **Intrinsic Curiosity Module (ICM)**: Introduces an intrinsic reward to encourage exploration in sparse reward environments.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/intrinsic-curiosity-module.git](https://github.com/alighasemi78/Curiosity-driven-Exploration-by-Self-supervised-Prediction.git)
   cd Curiosity-driven-Exploration-by-Self-supervised-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main python file:
```bash
python main.py
```

## Results

The project demonstrates the effectiveness of PPO + ICM in environments like MountainCar, where intrinsic rewards help the agent learn more effectively compared to PPO alone.
