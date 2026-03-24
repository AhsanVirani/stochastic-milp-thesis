# Stochastic Optimization Trading Strategies in Sequential Electricity Markets

This repository contains tools and models for analyzing energy markets, creating forecasts, and optimizing bid strategies. It includes two starter repositories: **dayahead** and **intraday**, which focus on day-ahead and intraday market models, respectively.

## Getting Started

### 1. Set Up the Environment
1. Install `virtualenv` if not already installed:
``` pip install virtualenv ```


2. Create a virtual environment:
  
3. Activate the virtual environment:
- On Linux/Mac:
  ```
  source venv/bin/activate
  ```
- On Windows:
  ```
  .\venv\Scripts\activate
  ```
4. Install all required libraries from `requirements.txt`:
``` pip install -r requirements.txt```

### 2. Repository Structure

The project is organized as follows:

```text
root/
├── dayahead/
│   ├── notebooks/
│   │   ├── StylizedFacts.ipynb
│   │   ├── ARIMAXModel.ipynb
│   │   └── ScenarioReduction.ipynb
│   ├── scenarios/da/
│   ├── reduced_scenarios/{cluster_name}/da/
│   ├── images/
├── intraday/
│   ├── notebooks/
│   │   ├── StylizedFacts.ipynb
│   │   ├── ARIMAXModel.ipynb
│   │   └── ScenarioReduction.ipynb
│   ├── scenarios/id/
│   ├── reduced_scenarios/{cluster_name}/id/
│   ├── images/
├── optimization/optim2/
├── images/scenario_trees/
├── Scenario Tree Representation.ipynb
└── Stochastic Optimization.ipynb
```

## Data Overview

Four main datasets are used in this project:

- **prices.csv**: Contains historical hourly market-clearing prices.
  - **DA_price**: Day-ahead market clearing price (MCP) per hour.
  - **ID3_price**: Intraday ID3 market clearing price per hour.
  - **delivery_begin**: Delivery start timestamp for each trading hour.

- **rl.csv**: Contains hourly residual load forecasts, captured 8 hours prior to delivery.
  - **rl**: Residual load forecast values (in MW).
  - **delivery_begin**: Forecast timestamp associated with each forecasted value.

- **HKZ_forecast_da.csv**: Contains the day-ahead generation forecast for the HKZ offshore wind park.
  - **frozen_fc_da**: Forecasted production values (in MW).
  - **Timestamp**: Forecast delivery timestamps (UTC time zone).

- **HKZ_actual_da.csv**: Contains the actual realized generation for the HKZ offshore wind park.
  - **rt_signal**: Real-time measured production output (in MW).
  - **Timestamp**: Actual delivery timestamps (UTC time zone).

---

These files serve as the foundation for:
- Building ARIMAX-based price forecasting models.
- Constructing stochastic scenario trees for price and production.
- Comparing forecasted vs. actual generation performance for HKZ.
- Optimizing bid strategies based on both market prices and production uncertainty.

Residual load forecasts (`rl.csv`) act as a major driver of electricity prices, especially under high renewable penetration.  
The HKZ production datasets (`HKZ_forecast_da.csv` and `HKZ_actual_da.csv`) enable the evaluation of forecast errors and provide the basis for realistic bid constraints in optimization models.


## Workflow

Before running the notebooks or optimization scripts, make sure to set up a `config.json` file in the root directory of the repository. This file specifies which cluster, model, and optimizer to use across the entire project.

### Example `config.json`
```json
{
  "cluster": "10",
  "model_name": "ARIMAX",
  "optimizer": "optim2"
}
```
### Day-Ahead Market Analysis

1. **Stylized Facts Analysis**  
Navigate to `dayahead/` and run `Stylized Facts About Electricity Prices.ipynb`. This notebook analyzes energy market stylized facts. Simply execute the notebook to view the analysis.
    - Save stylized fact images to `images/DA`.

2. **ARIMAX Model**  
Run `ARIMAX Model on Day Ahead Electricity Price & Scenario Tree.ipynb` to apply the ARIMAX model on the day-ahead market data. This will:
    - Save generated scenarios in `scenarios/da/`.
    - Save generated scenarios images to `images/DA/forecasts`.
    - Save model generated results and feature engineering images to `images/DA`.

3. **Scenario Reduction**  
Run `scenario Reduction Technique.ipynb` to reduce the generated scenarios using k-means clustering. This will:
    - Save reduced scenarios in `reduced_scenarios/{cluster_name}/da/`.
    - Save result images in `images/DA/forecasts`.

The `{cluster_name}` corresponds to the number of clusters defined in the configuration file (e.g., 5, 10, or 25). For example:
- Cluster size 5 → Saved in `C5`.
- Cluster size 10 → Saved in `C10`.
- Cluster size 25 → Saved in `C25`.

this comes from the config.json as setup on top. Varying cluster sizes can be configured and tested by changing this number.

**Important:** The day-ahead files must be completed before proceeding with the intraday files.

### Intraday Market Analysis

Follow the same steps as above for the intraday market by running the notebooks in `intraday/` in this order:

1. **Stylized Facts Analysis**  
Run `Stylized Facts About Electricity Prices.ipynb` to analyze stylized facts of intraday market prices and volumes.

2. **ARIMAX Model**  
Run `ARIMAX Model on Day Ahead Electricity Price & Scenario Tree.ipynb` to apply the ARIMAX model on intraday data. 

2. **Scenario Reduction**  
Run `scenario Reduction Technique.ipynb` to apply the ARIMAX model on intraday data.

Similar to day-ahead, all the intraday results are saved in 
- Save result images in the `images/ID` directory.
- Save generated scenarios in `scenarios/id/`.
- Save generated reduced scenarios in `reduced_scenarios/{cluster_name}/id/`.

### Scenario Tree Representation

Once all day-ahead and intraday workflows are completed, you can visualize scenario trees by running the notebook at the root level:  
`Scenario Tree Representation.ipynb`.  

This will save scenario tree images in:  
`images/scenario_trees/{cluster_name}`.

## Optimization: Creating Bid Ladders

### Gurobi Optimizer Setup

The optimization process uses Gurobi for solving stochastic programming problems. For the purposes of this thesis, we use licensed version of Gurobi provided by Vattenfall Energy Trading to run the complex Mixed Integer Linear Problem. The optimizer that we use is located in `optimization/optim2/`.

1. Place your Gurobi license credentials in a file named `gurobi.lic` with this format:
    ```
    # Gurobi Cluster Manager license file
    # Your credentials are private and should not be shared or copied to public repositories.
    CSMANAGER=https://your-cluster-manager-url.com:61080
    CSAPIACCESSID=your-access-id-placeholder
    CSAPISECRET=your-secret-key-placeholder
    CSAPPNAME=your-app-name-placeholder
    ```

2. Run the optimization notebook:  
`Stochastic Optimization.ipynb`.  

This notebook performs stochastic optimization to generate bid ladders and saves results in:  
`strategy/optim2/gurobi/schedule/{cluster_name}/{strategy_name}`.

### Results

- **Scenarios** are saved in:
  - `dayahead/scenarios/da/` and `dayahead/reduced_scenarios/{cluster_name}/da/`
  - `intraday/scenarios/id/` and `intraday/reduced_scenarios/{cluster_name}/id/`

- **Optimized results** are saved in:
  - `strategy/optim2/gurobi/schedule/{cluster_name}/{strategy_name}/{date}.csv`  
    where `{date}` is the day for which optimization is performed.

---

## Summary of Outputs

- **Images:** All visualizations (e.g., model fits, scenario plots, power curves) are saved in the corresponding `images/` directories under `DA/` and `ID/`.

- **Scenarios:** 
  - Raw generated scenarios → `/scenarios/` 
  - Reduced scenarios → `/reduced_scenarios/{cluster_name}/`

- **Optimization Results:**
    The optimizer that we use for the thesis is located at `optimization/optim2/StochasticOptim_Gurobi.py`
  - Final bidding schedules and revenues → `/strategy/optim2/gurobi/schedule/{cluster_name}/{strategy_name}/`
  and the corresponding images to `/images/strategy/optim2/gurobi`

### Strategy Comparison and Benchmarking

After running the optimization notebooks and generating strategy results, you can compare all strategies against the **Vattenfall benchmark** using:

#### `Optimizer Results Across Strategy Comparison.ipynb`

This notebook:
- Loads optimized results from `/strategy/optim2/gurobi/schedule/{cluster_name}/{strategy_name}/`
- Compares each strategy's revenue to the Vattenfall benchmark (e.g., 90% DA sale)
- Computes and visualizes:
  - **Excess returns** (percentage gain/loss over benchmark)
  - **Cumulative excess returns** across dates

Make sure all relevant strategy schedules are generated before running this notebook. It serves as the final evaluation step for analyzing which bidding strategies outperform the benchmark.

