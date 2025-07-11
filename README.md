# Counterfactual Explanations for Time Series using Reinforcement Learning

This repository implements a system for generating **counterfactual explanations** for time series classification tasks using **Deep Reinforcement Learning (DRL)**. Counterfactual explanations are a powerful tool in **Explainable AI (XAI)**, providing insight into model predictions by identifying how small input changes could alter the predicted outcome.

Unlike traditional optimization-based methods—which can be computationally expensive and inflexible—this approach formulates the generation of counterfactuals as a **sequential decision-making process**. An RL agent learns to apply a **binary mask** over the input time series, guided by comparisons with its **Nearest Unlike Neighbor (NUN)**, in order to generate minimal yet effective transformations that flip the model prediction while preserving plausibility.

## Key Features

- Supports **univariate time series** from the UCR archive
- Integrates black-box classifiers and reinforcement learning agents
- Produces interpretable, sparse, and plausible counterfactual explanations
- Includes a modular and extensible codebase for experimentation
- Tracks key evaluation metrics for effectiveness and interpretability

## Repository Structure

The repository is organized as follows:

    cfe_rl/
    ├── data/ # Raw and preprocessed datasets
    │ └── UCR/ # Includes Chinatown, ECG200, FordA, GunPoint, Beef
    ├── models/ # Trained classification models
    ├── params/ # Agent and model configuration files (JSON)
    │ ├── agents/
    │ └── models/
    ├── results/ # Logs, counterfactual outputs and evaluation metrics
    ├── src/ # Core system logic
    │ ├── agents/ # Agent training, policy architectures, callbacks
    │ ├── data/ # Dataset loading, scaling and NUN computation
    │ ├── environments/ # Custom Gymnasium environment for counterfactuals
    │ ├── models/ # FCN classifiers and RL architectures
    │ └── utils/ # Logging, metrics, loss functions, helper utilities
    ├── experiments/ # Notebooks, exploratory scripts and logs
    │ ├── notebooks/
    │ ├── scripts/
    │ └── logs/
    ├── fcn.py # Script to train FCNs for time series classification
    ├── main.py # Main entry point for RL experiment execution
    └── requirements.txt # Python dependencies


## Execution Flow

The typical usage of this repository consists of the following steps:

1. **Train a classifier**  
   Run `fcn.py` to train a Fully Convolutional Network on a given UCR dataset. The resulting model is stored under `models/`.

2. **Configure an RL agent**  
   Edit the corresponding JSON file in `params/agents/` to define the agent's architecture and training settings.

3. **Run the RL experiment**  
   Use `main.py` to launch training, which sets up the environment, initializes the agent and performs training and evaluation. Example:
   ```bash
   python main.py --config params/agents/ppo_chinatown.json

4. **Evaluate results**
Generated counterfactuals, performance metrics, and logs are stored in `results/`.

## Summary of the Method

  Counterfactual explanations help interpret the decisions of black-box models by revealing how small changes to input data could result in different predictions. Most existing approaches rely on gradient-based optimization, which can be slow and brittle—especially with sequential data.

This work introduces a novel method for generating counterfactuals in time series classification by training a reinforcement learning agent that learns class-specific transformation policies. By interacting with an input time series and its NUN, the agent modifies a binary mask that blends values between the original and its counterpart, aiming to change the model's prediction while maintaining fidelity to the input distribution.

This approach is applied to univariate time series datasets, training a separate agent per class. The method introduces new evaluation metrics tailored to the sequential setting, and emphasizes interpretability, sparsity, and plausibility of the generated examples
