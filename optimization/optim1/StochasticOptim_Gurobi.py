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

        # Load forecasts and actual production
        self.forecasted_generation = pd.read_csv(f"data/HKZ_forecast_da.csv", index_col=0).loc[
            f"{date} 00:00:00":f"{date} 23:30:00", :
        ]['frozen_fc_da'].values
        self.actual_production = pd.read_csv(f"data/HKZ_actual_prod.csv", index_col=0).loc[
            f"{date} 00:00:00":f"{date} 23:30:00", :
        ]['rt_signal'].values

    def add_forecast_constraints(self, model, DA_bids, ID_bids):
        for h in range(24):
            model.addConstr(
                gp.quicksum(DA_bids[i, h] for i in range(self.cluster_size)) + 
                gp.quicksum(ID_bids[i, j, h] for i in range(self.cluster_size) for j in range(self.cluster_size)) <= (self.forecasted_generation[h] + 100),
                name=f"forecast_constraint_h{h}"
            )

    def add_min_da_constraints(self, model, threshold, DA_bids):
        for h in range(24):
            model.addConstr(
                gp.quicksum(DA_bids[i, h] for i in range(self.cluster_size)) >= threshold * self.forecasted_generation[h],
                name=f"min_da_bids_h{h}"
            )

    def add_cvar_constraints(self, model, contribution_margin, eta, s):
        for i in range(self.cluster_size):
            model.addConstr(s[i] >= eta - contribution_margin[i], name=f"cvar_constraint_s{i}")

    def add_non_anticipativity_constraints(self, model, DA_bids, ID_bids):
        for h in range(24):
            if min(self.da[:, h]) >= self.limit_price:  # Only enforce if all prices are above the limit price
                for i in range(self.cluster_size):
                    model.addConstr(
                        DA_bids[i, h] == DA_bids[0, h],
                        name=f"non_anticipativity_DA_{i}_{h}"
                    )

        for h in range(24):
            if min(self.id_li[:, :, h].flatten()) >= self.limit_price:  # Only enforce if all prices are above the limit price
                for i in range(1, self.cluster_size):
                    for j in range(self.cluster_size):
                        model.addConstr(
                            ID_bids[i, j, h] == ID_bids[0, j, h],
                            name=f"non_anticipativity_ID_{i}_{j}_{h}"
                        )

    def add_bidding_limit_constraints(self, model, DA_bids, ID_bids):
        for i in range(self.cluster_size):  # Scenarios
            for h in range(24):  # Hours
                if self.da[i, h] <= self.limit_price:
                    model.addConstr(
                        DA_bids[i, h] == 0, name=f"DA_no_bid_below_limit_{i}_{h}"
                    )

        for i in range(self.cluster_size):  # Day-ahead scenarios
            for j in range(self.cluster_size):  # Intraday scenarios
                for h in range(24):  # Hours
                    if self.id_li[i, j, h] <= self.limit_price:
                        model.addConstr(
                            ID_bids[i, j, h] == 0, name=f"ID_no_bid_below_limit_{i}_{j}_{h}"
                        )

    def create_model(self):
        model = gp.Model(env=env)

        # Decision variables
        DA_bids = model.addVars(self.cluster_size, 24, lb=0, name="DA_bids")
        ID_bids = model.addVars(self.cluster_size, self.cluster_size, 24, lb=0, name="ID_bids")
        eta = model.addVar(lb=0, name="eta")
        s = model.addVars(self.cluster_size, lb=0, name="s")

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
        cvar_term = gp.quicksum(
            self.da_probab[i] * self.id_probab[i, j] * s[i] for i in range(self.cluster_size) for j in range(self.cluster_size)
        )
        cvar = eta - (1 / (1 - self.alpha_cvar)) * cvar_term

        # Set objective
        model.setObjective((1 - self.lambda_risk_aversion) * expected_contribution_margin + self.lambda_risk_aversion * cvar, GRB.MAXIMIZE)

        # Add constraints using modularized methods
        self.add_forecast_constraints(model, DA_bids, ID_bids)
        self.add_min_da_constraints(model, 0.5, DA_bids)
        self.add_cvar_constraints(model, contribution_margin, eta, s)
        self.add_non_anticipativity_constraints(model, DA_bids, ID_bids)
        self.add_bidding_limit_constraints(model, DA_bids, ID_bids)

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
