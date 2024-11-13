# A Mathematical Certification for Positivity Conditions in Neural Networks with Applications to Partial Monotonicity and Ethical AI

## Overview
This repository contains the code and data supporting the paper titled ["A Mathematical Certification for Positivity Conditions in Neural Networks with Applications to Partial Monotonicity and Trustworthy AI"](https://arxiv.org/abs/2406.08525). The repository is organized to facilitate the replication of results and to provide resources for further exploration of the concepts discussed in the paper.


## Repository Structure

The repository is structured as follows:

- **Notebooks/**: Contains Jupyter notebooks for the case studies presented in the paper.
  - `ESL_DATASET.ipynb`: Case study on the ESL dataset.
  - `Heat_Equation_Colab.ipynb`: Case study on the heat equation.
  - **Notebooks/Models/**: Directory to store the trained models.
    - `ESL_Dataset.pt`: Checkpoint file for the trained model.

- **Scripts/**: Contains scripts with the core code used in the paper, including the implementation of the LipVor algorithm.
  - `LipVor_functions.py`: Implementation of the LipVor algorithm and the needed python functions.
  - `LipVor_functions.py`: Implementation of the LipVor certification tool. An example of use can be seen in the `ESL_DATASET.ipynb`
  - `MonoNN.py`: Implementation of the Monotonic Neural Network class.
  - `generate_data.py`: Script to generate data for the case studies.
  - `train.py`: Script to train the neural network models.
  - `Utilities.py`: Utility functions used throughout the project.



## Getting Started
To use the LipVor algorithm, follow these steps:

1. Clone this repository: `git clone https://github.com/alejandropolo/LipschitzNN.git`
2. Install the required dependencies by creating the environment from the `environment.yaml` file: 
   ```bash
   conda env create -f environment.yaml
3. Run the case study notebooks.
4. Feel free to explore the code and customize it according to your specific requirements.

## Example: Using the LipVorCertification Function

The `LipVorCertification` function in `LipVor_Certification.py` is used to certify a neural network model using the LipVor algorithm. Below is an example of how to use this function. Additionally, an example is provided in the `ESL_DATASET.ipynb` to demonstrate its application in a real-world scenario.

![Example of LipVorCertification execution](Figures/LipVor_exec_example.png)


```python
import torch
from LipVor_Certification import LipVorCertification

# Define your neural network model
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNN()

# Define the activation function, weights, and biases
actfunc = 'relu'
weights = [param.data.numpy() for param in model.parameters() if param.requires_grad and param.ndimension() == 2]
biases = [param.data.numpy() for param in model.parameters() if param.requires_grad and param.ndimension() == 1]

# Define the monotone relations for each variable
monotone_relations = [1, -1]  # Example: first variable is increasing, second is decreasing

# Define the global Lipschitz constant
global_lipschitz_constant = 1.0

# Load the training data tensor
X_train_tensor = torch.load('Data/X_train_data.pt')

# Call the LipVorCertification function
LipVorCertification(
    model=model,
    actfunc=actfunc,
    weights=weights,
    biases=biases,
    monotone_relations=monotone_relations,
    variable_index=list(range(X_train_tensor.shape[1])),
    global_lipschitz_constant=global_lipschitz_constant,
    X_train_tensor=X_train_tensor,
    n_initial_points=10,
    max_iterations=100,
    epsilon_derivative=0.1,
    epsilon_proyection=0.01,
    probability=0.1,
    seed=1
)
