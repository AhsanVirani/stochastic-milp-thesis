import sys
sys.path.append('../..')

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

gurobi_dict = {}
with open('gurobi.lic', 'r') as file:
    for line in file:
        if line.strip() and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            gurobi_dict[key] = value

# Create an environment with your WLS license
params = {
    "CSMANAGER"    : gurobi_dict['CSMANAGER'],
    "CSAPIACCESSID": gurobi_dict['CSAPIACCESSID'],
    "CSAPISECRET"  : gurobi_dict['CSAPISECRET'], 
    "CSAPPNAME"    : gurobi_dict['CSAPPNAME']
}
env = gp.Env(params=params)
print(params)

class BiddingOptimization:
    def __init__(self, limit_price=-2, lambda_risk_aversion=1, alpha_cvar=0.95, cluster_size=5):
        self.limit_price = limit_price
        self.lambda_risk_aversion = lambda_risk_aversion
        self.alpha_cvar = alpha_cvar
        self.cluster_size = cluster_size

    @staticmethod
    def load_csv(file_path):
        data = pd.read_csv(file_path).iloc[:, 1:]
        return data.to_numpy()

    def load_data(self, date):
        self.date = date
        
        # Load day-ahead data and probabilities
        self.da = self.load_csv(f"reduced_scenarios/C{self.cluster_size}/da/scenario_{date}.csv")
        self.da_probab = np.loadtxt(f"reduced_scenarios/C{self.cluster_size}/da/probability_{date}.txt")

        # Load intraday scenarios and probabilities
        self.id_li = np.array([
            self.load_csv(f"reduced_scenarios/C{self.cluster_size}/id/scenario_{i}_{date}.csv")
            for i in range(self.cluster_size)
        ])
        self.id_probab = np.array([
            np.loadtxt(f"reduced_scenarios/C{self.cluster_size}/id/probability_{i}_{date}.txt")
            for i in range(self.cluster_size)
        ])

        # normalize probabilities to 1
        self.da_probab /= np.sum(self.da_probab)
        self.da_probab /= np.sum(self.da_probab)
        
        # Load forecasts and actual production
        self.forecasted_generation = 0.80 * pd.read_csv(f"data/HKZ_forecast_da.csv", index_col=0).loc[
            f"{date} 00:00:00":f"{date} 23:30:00", :
        ]['frozen_fc_da'].values
    
    def add_forecast_constraints(self, model, min_threshold, max_threshold, DA_bids, ID_bids):
        """
        Adds constraints to ensure that the day-ahead (DA) and intraday (ID) bids for each hour 
        respect the forecasted generation limits and do not exceed actual production.
    
        Parameters:
        - model: Gurobi model object.
        - threshold: Minimum fraction of forecasted generation that must be committed to bids (e.g., 0.5 for 50%).
        - DA_bids: Gurobi decision variables for day-ahead bids.
        - ID_bids: Gurobi decision variables for intraday bids.
    
        Ensure total DA and ID bids for each hour and scenario are at least `threshold * forecasted_generation[h]`.
        These constraints help manage volume risk, ensuring sufficient DA commitment while avoiding overcommitting resources.
        """
        for h in range(24):  # Iterate over hours
            # Minimum commitment constraint based on forecasted generation
            model.addConstr(
                gp.quicksum(DA_bids[i, h] for i in range(self.cluster_size)) +
                gp.quicksum(ID_bids[i, j, h] for i in range(self.cluster_size) for j in range(self.cluster_size)) >= min_threshold * self.forecasted_generation[h],
                name=f"min_forecast_constraint_hour{h}"
            )

            model.addConstr(
                gp.quicksum(DA_bids[i, h] for i in range(self.cluster_size)) +
                gp.quicksum(ID_bids[i, j, h] for i in range(self.cluster_size) for j in range(self.cluster_size)) <= max_threshold * self.forecasted_generation[h],
                name=f"max_forecast_constraint_hour{h}"
            )
    
    def add_da_threshold_constraints(self, model, min_threshold, max_threshold, DA_bids):
        """
        Adjusted threshold constraints for DA bids:
        - Enforces bids to be zero when forecast generation is zero.
        - Enforces bids to be within the specified percentage thresholds when forecast generation is positive.
        """
        for h in range(24):  # Iterate over hours
            forecast = self.forecasted_generation[h]
            print(f"Hour {h}: Forecasted Generation = {forecast}")
        
            if forecast > 0:  # Forecast generation is positive
                # Ensure total DA bids are at least the minimum threshold of the forecasted generation
                model.addConstr(
                    gp.quicksum(DA_bids[i, h] for i in range(self.cluster_size)) >= min_threshold * forecast,
                    name=f"min_da_constraint_hour{h}"
                )
                # Ensure total DA bids do not exceed the maximum threshold of the forecasted generation
                model.addConstr(
                    gp.quicksum(DA_bids[i, h] for i in range(self.cluster_size)) <= max_threshold * forecast,
                    name=f"max_da_constraint_hour{h}"
                )
            else:  # Forecast generation is zero
                # Explicitly set all DA bids to zero
                for i in range(self.cluster_size):
                    model.addConstr(DA_bids[i, h] == 0, name=f"DA_zero_bid_gen0_{i}_{h}")

    
    def add_cvar_constraints(self, model, contribution_margin, eta, s):
        """
        Adds constraints to enforce the Conditional Value at Risk (CVaR) calculation.

        Parameters:
        - model: Gurobi model object.
        - contribution_margin: List of contribution margin expressions for each scenario.
        - eta: Gurobi variable representing the CVaR threshold.
        - s: Gurobi variables representing shortfall variables for each scenario.

        Constraints:
        s[i] >= eta - contribution_margin[i]

        This ensures that the shortfall variables (s[i]) represent the deviation
        of the contribution margin from the CVaR threshold (eta).
        """
        for i in range(self.cluster_size):
            model.addConstr(s[i] >= eta - contribution_margin[i], name=f"cvar_constraint_s{i}")

    def add_non_anticipativity_constraints(self, model, DA_bids, ID_bids, tolerance=0.05):
        """
        Adds non-anticipativity constraints to enforce consistent bidding decisions 
        across scenarios for the day-ahead and intraday markets only when prices 
        are within a given tolerance.
    
        Parameters:
        - model: Gurobi model object.
        - DA_bids: Gurobi decision variables for day-ahead bids.
        - ID_bids: Gurobi decision variables for intraday bids.
        - tolerance: Float, percentage tolerance within which prices are considered similar.
        """
        # Non-anticipativity for day-ahead bids
        for h in range(24):  # Iterate over hours
            for i in range(1, self.cluster_size):  # Compare all scenarios with scenario 0
                if abs(self.da[i, h] - self.da[0, h]) / max(abs(self.da[0, h]), 1e-6) <= tolerance:  # Within tolerance
                    model.addConstr(
                        DA_bids[i, h] == DA_bids[0, h],
                        name=f"non_anticipativity_DA_{i}_{h}"
                    )
    
        # Non-anticipativity for intraday bids
        for h in range(24):  # Iterate over hours
            for i in range(1, self.cluster_size):  # Compare all scenarios with scenario 0
                for j in range(self.cluster_size):  # Compare intraday sub-scenarios
                    if abs(self.id_li[i, j, h] - self.id_li[0, j, h]) / max(abs(self.id_li[0, j, h]), 1e-6) <= tolerance:  # Within tolerance
                        model.addConstr(
                            ID_bids[i, j, h] == ID_bids[0, j, h],
                            name=f"non_anticipativity_ID_{i}_{j}_{h}"
                        )

                        
    def add_bidding_limit_constraints(self, model, DA_bids, ID_bids):
        """
        Unified constraints for DA and ID bids:
        - Ensures no positive sell bids when prices are below or equal to the limit price.
        - Explicitly sets all bids to zero if forecasted generation is zero.
        """
        for h in range(24):  # Iterate over hours
            forecast = self.forecasted_generation[h]
            
            if forecast == 0:  # If forecast generation is zero
                for i in range(self.cluster_size):
                    # Explicitly set day-ahead bids to zero
                    model.addConstr(DA_bids[i, h] == 0, name=f"DA_zero_bid_gen0_{i}_{h}")
                    for j in range(self.cluster_size):
                        # Explicitly set intraday bids to zero
                        model.addConstr(ID_bids[i, j, h] == 0, name=f"ID_zero_bid_gen0_{i}_{j}_{h}")
            else:  # Forecast generation is non-zero
                for i in range(self.cluster_size):
                    # If the day-ahead price is less than or equal to the limit price
                    if self.da[i, h] <= self.limit_price:
                        # Explicitly set day-ahead bids to zero
                        model.addConstr(DA_bids[i, h] == 0, name=f"DA_zero_bid_limit_{i}_{h}")
                    for j in range(self.cluster_size):
                        # If the intraday price is less than or equal to the limit price
                        if self.id_li[i, j, h] <= self.limit_price:
                            # Explicitly set intraday bids to zero
                            model.addConstr(ID_bids[i, j, h] <= 0, name=f"ID_no_positive_sell_{i}_{j}_{h}")


    def create_model(self):
        model = gp.Model(env=env)

        # DA_bids are non negative - we have generation to sell
        DA_bids = model.addVars(range(self.cluster_size), 24, 
                                lb=0,
                                ub={(i, h): 1.1 * self.forecasted_generation[h] for i in range(self.cluster_size) for h in range(24)},
                                name="DA_bids")
        # ID_bids are real 
        ID_bids = model.addVars(range(self.cluster_size), range(self.cluster_size), 24, 
                                lb={(i, j, h): -1.1 * self.forecasted_generation[h] for i in range(self.cluster_size) for j in range(self.cluster_size) for h in range(24)},
                                ub={(i, j, h): 1.1 * self.forecasted_generation[h] for i in range(self.cluster_size) for j in range(self.cluster_size) for h in range(24)},
                                name="ID_bids")


        eta = model.addVar(lb=0, name="eta")
        s = model.addVars(range(self.cluster_size), lb=0, name="s")
            
        # Objective function components
        DA_revenue = [gp.quicksum(self.da[i, h] * DA_bids[i, h] for h in range(24)) for i in range(self.cluster_size)]
        ID_revenue = [
            gp.quicksum(
                gp.quicksum(self.id_li[i, j, h] * ID_bids[i, j, h] * self.id_probab[i, j] for h in range(24)) for j in range(self.cluster_size)
            )
            for i in range(self.cluster_size)
        ]
        total_revenue = [DA_revenue[i] + ID_revenue[i] for i in range(self.cluster_size)]
        total_cost = [
            gp.quicksum(self.limit_price * DA_bids[i, h] for h in range(24)) +
            gp.quicksum(self.limit_price * ID_bids[i, j, h] * self.id_probab[i, j] for j in range(self.cluster_size) for h in range(24))
            for i in range(self.cluster_size)
        ]
        contribution_margin = [total_revenue[i] - total_cost[i] for i in range(self.cluster_size)]
    
        # Expected contribution margin and CVaR term
        expected_contribution_margin = gp.quicksum(self.da_probab[i] * contribution_margin[i] for i in range(self.cluster_size))
        cvar_term = gp.quicksum(self.da_probab[i] * s[i] for i in range(self.cluster_size))
        cvar = eta - (1 / (1 - self.alpha_cvar)) * cvar_term
                
        penalty_weight = 0.05

        # Updated objective with price-weighted penalties
        model.setObjective(
            (1 - self.lambda_risk_aversion) * expected_contribution_margin +
            self.lambda_risk_aversion * cvar -
            penalty_weight * (
                gp.quicksum((DA_bids[i, h] - self.forecasted_generation[h])**2 for i in range(self.cluster_size) for h in range(24)) +
                gp.quicksum((ID_bids[i, j, h] - self.forecasted_generation[h])**2 for i in range(self.cluster_size) for j in range(self.cluster_size) for h in range(24))
            ),
            GRB.MAXIMIZE
        )
        
        # Add constraints
        self.add_forecast_constraints(model, 0.9, 1.1, DA_bids, ID_bids)
        self.add_da_threshold_constraints(model, 0.5, 1.2, DA_bids)
        self.add_cvar_constraints(model, contribution_margin, eta, s)
        self.add_non_anticipativity_constraints(model, DA_bids, ID_bids)
        # self.add_bidding_limit_constraints(model, DA_bids, ID_bids)
        
        self.model = model
        self.DA_bids = DA_bids
        self.ID_bids = ID_bids
        self.eta = eta

    def solve(self):
        # Set Gurobi parameters
        self.model.setParam('TimeLimit', 300)
        self.model.setParam('MIPGap', 0.01)
        self.model.setParam('Threads', 4)
        self.model.setParam('Presolve', 1)

        # Solve the model
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimization successful for {self.date}!")
            self.DA_bids_optimal = np.array([
                [self.DA_bids[i, h].X for h in range(24)] for i in range(self.cluster_size)
            ])
            self.ID_bids_optimal = np.array([
                [[self.ID_bids[i, j, h].X for h in range(24)] for j in range(self.cluster_size)] for i in range(self.cluster_size)
            ])
        else:
            print(f"Optimization failed for {self.date}!")

    def run(self, dates):
        results = {}
        for date in dates:
            print(f"Running optimization for {date}...")
            self.load_data(date)
            self.create_model()
            self.solve()
            results[date] = {
                "DA_bids": self.DA_bids_optimal,
                "ID_bids": self.ID_bids_optimal,
            }

        return results
