# Counterfactual Explanations for Time Series Project

This is a first approximation of the project. This repository implements a system to generate counterfactual explanations of a time series classification model, by searching for the Nearest Unlike Neighbor (NUN) and using a binary mask that exchanges values between the original sample and its NUN.


> [!IMPORTANT] 
> For the moment, this README focuses on describing the project directory structure.

## Project Structure

The following folder structure organizes the project in a modular and scalable way:

cfe_rl/
├── data/                    
   └── UCR/
       ├── chinatown/
       ├── ecg200/
       └── forda/
...


## Description of Folders

- **data/**: Contains the data sets, both in their raw and preprocessed form.  
- **models/**: Stores the trained models along with their architectures. They are organized in subdirectories by dataset and architecture.  
- **params/**: It gathers the configuration parameters for the models and the RL agents.  
- **src/**: It is the core of the project code.  
  - **environments/**: Definition of the RL environment, which includes the counterfactual explanation generation logic.  
  - **agents/**: Implementations of the RL agents in charge of modifying the binary mask.  
  - **data/**: Code for data handling, loading and preprocessing (e.g. `data_manager.py`).  
  - **models/**: Functions and classes to interact with the classification models.  
  - **utils/**: Utility functions shared by other modules.  
  - **main.py**: Entry point to run experiments or training.  
- **experiments/**: Space for experimentation, including notebooks, scripts and logs.  
- **results/**: Stores the results of runs and analysis (e.g., counterfactual explanations and metrics).  
- **docs/**: Documentation, reports and project design notes.  
- **requirements.txt**: List of dependencies required to run the project.

## Next Steps