
# Optimizing the Power consumptions:

I have created a notebook called `optimize_power_consumption.ipynb` where you get to see how an individual can solve an optimization problem in two ways.
- Machine Learning Approach
- Mathematical Modeling Approach
### Step(1): Machine Learning Approach
### Linear Regression with PuLP for Optimizing Power Consumption

### Problem Overview

This project solves an optimization problem that combines **Linear Regression** for predicting the power consumption of machines based on their production rate (GPH) and **PuLP** for minimizing the total power consumption of 20 machines in a factory. The key objectives and constraints are as follows:

- The factory must produce exactly **9000 units of goods per hour (GPH)** using 20 machines.
- There are **two types of machines**:
  - **Machine Type 1**: Each machine can produce between **180 and 600 GPH**.
  - **Machine Type 2**: Each machine can produce between **300 and 1000 GPH**.
  
#### Workflow:
1. **Linear Regression**: Build a predictive model to estimate the power consumption of each machine type as a function of GPH based on training data.
2. **PuLP Optimization**: Use the predicted power functions in a linear optimization framework to minimize the total power consumption of the factory while meeting the production target (9000 GPH).

### Objective Function

The objective is to minimize total power consumption:

$$
P_{\text{total}} = \sum_{i=1}^{10} \hat{P}_1(GPH_i) + \sum_{i=11}^{20} \hat{P}_2(GPH_i)
$$

Where:
- \(\hat{P}_1(GPH_i)\) is the predicted power consumption for **Machine Type 1**, learned from the Linear Regression model.
- \(\hat{P}_2(GPH_i)\) is the predicted power consumption for **Machine Type 2**, also learned from the Linear Regression model.

### Constraints

1. **Total GPH**: The sum of GPH across all machines must equal 9000.

$$
\sum_{i=1}^{20} GPH_i = 9000
$$

2. **GPH Bounds**:
   - For **Machine Type 1**: \(180 \leq GPH_i \leq 600\)
   - For **Machine Type 2**: \(300 \leq GPH_i \leq 1000\)

###-----------
### Step(2): Mathematical Modeling 

## Gurobi-Based Non-Linear Optimization for Minimizing Power Consumption

### Problem Overview

This project solves an optimization problem where the goal is to minimize the total **power consumption** of 20 machines operating in a factory, subject to the following constraints:
- **Goods Per Hour (GPH)**: The factory must produce a total of 9000 units of goods per hour using all machines combined.
- **Two types of machines**:
  - **Machine Type 1**: Each machine can produce between **180 and 600 GPH**.
  - **Machine Type 2**: Each machine can produce between **300 and 1000 GPH**.
- **Power Consumption Function**: The power consumption of each machine is modeled as a **quadratic function** of the GPH:
  - Machine Type 1: \( P_1(GPH) = 0.005 \times GPH^2 + 1.2 \times GPH + 100 \)
  - Machine Type 2: \( P_2(GPH) = 0.004 \times GPH^2 + 1.1 \times GPH + 150 \)

## Gurobi Code

Here is the Gurobi code for solving the optimization problem:

```python
from gurobipy import Model, GRB

# Create a model
m = Model()

# Define variables with bounds
gph_vars = m.addVars(20, lb=[180]*10 + [300]*10, ub=[600]*10 + [1000]*10, name="gph")

# Define objective function (quadratic)
m.setObjective(
    sum(0.005 * gph_vars[i]**2 + 1.2 * gph_vars[i] + 100 for i in range(10)) +
    sum(0.004 * gph_vars[i]**2 + 1.1 * gph_vars[i] + 150 for i in range(10, 20)),
    GRB.MINIMIZE
)

# Add constraint: total GPH must be 9000
m.addConstr(sum(gph_vars[i] for i in range(20)) == 9000)

# Solve the model
m.optimize()

# Output results
optimal_gph = [gph_vars[i].x for i in range(20)]
total_optimal_power = m.objVal

print("Optimal GPH values:", optimal_gph)
print("Total Power Consumption:", total_optimal_power)
```

- Minimizes the total power consumption.
- Meets the GPH constraint of producing exactly 9000 units per hour.

## Solution Overview

We use **Gurobi**, a powerful commercial optimization solver, to solve this non-linear optimization problem. The problem is formulated as follows:

- **Decision variables**: GPH for each machine.
- **Objective function**: Minimize the sum of the power consumption across all machines.
- **Constraints**:
  - The sum of the GPH for all machines must equal 9000.
  - Each machine's GPH must be within its specified bounds.

The Gurobi solver finds the optimal solution by minimizing the quadratic objective function while respecting the constraints.

## Files in the Repository

- `optimize_power_consumption.ipynb`: The Python script containing the Gurobi code to solve the optimization problem.
- `README.md`: This file, which explains how to run the code and provides details on the problem and solution.
- `requirements.txt`: Lists the required Python packages (including Gurobi).

## Dependencies

Before running the code, ensure you have the following installed:

- **Python 3.x**
- **Gurobi**: You need to have Gurobi installed and a valid Gurobi license. If you're using Gurobi for academic purposes, you can get a free academic license from the [Gurobi website](https://www.gurobi.com).

You can install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```
The requirements.txt file should contain the following dependencies:
Explanation:
- **gurobipy:** Python library for Gurobi optimization.
- **pulp:** Python library for linear programming and optimization.
- **scikit-learn:** Machine learning library for regression, classification, and more.
- **seaborn:** Visualization library for statistical graphics based on Matplotlib.
- **matplotlib:** Library for creating static, animated, and interactive visualizations.
- **numpy:** Library for numerical operations and array handling.