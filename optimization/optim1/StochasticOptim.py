import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from docplex.mp.environment import Environment

env = Environment()
env.cplex_path = r"C:\Program Files\IBM\ILOG\CPLEX_Studio\version\cplex\bin\x64_win64\cplex.exe"

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

    def add_forecast_constraints(self, mdl, DA_bids, ID_bids):
        for h in range(24):
            mdl.add_constraint(
                mdl.sum(DA_bids[i, h] for i in range(self.cluster_size)) + \
                mdl.sum(ID_bids[i, j, h] for i in range(self.cluster_size) for j in range(self.cluster_size)) <= (self.forecasted_generation[h] + 100),
                f"forecast_constraint_h{h}"
            )

    def add_min_da_constraints(self, mdl, threshold, DA_bids):
        for h in range(24):
            mdl.add_constraint(
                mdl.sum(DA_bids[i, h] for i in range(self.cluster_size)) >= threshold * self.forecasted_generation[h],
                f"min_da_bids_h{h}"
            )

    def add_cvar_constraints(self, mdl, contribution_margin, eta, s):
        for i in range(self.cluster_size):
            mdl.add_constraint(s[i] >= eta - contribution_margin[i], f"cvar_constraint_s{i}")

    def add_non_anticipativity_constraints(self, mdl, DA_bids, ID_bids):
        for h in range(24):
            if min(self.da[:, h]) >= self.limit_price:  # Only enforce if all prices are above the limit price
                for i in range(self.cluster_size):
                    mdl.add_constraint(
                        DA_bids[i, h] == DA_bids[0, h],
                        f"non_anticipativity_DA_{i}_{h}"
                    )

        for h in range(24):
            if min(self.id_li[:, :, h].flatten()) >= self.limit_price:  # Only enforce if all prices are above the limit price
                for i in range(self.cluster_size):
                    for j in range(self.cluster_size):
                        mdl.add_constraint(
                            ID_bids[i, j, h] == ID_bids[0, j, h],
                            f"non_anticipativity_ID_{i}_{j}_{h}"
                        )

    def add_bidding_limit_constraints(self, mdl, DA_bids, ID_bids):
        for i in range(self.cluster_size):  # Scenarios
            for h in range(24):  # Hours
                if self.da[i, h] <= self.limit_price:
                    mdl.add_constraint(
                        DA_bids[i, h] == 0, f"DA_no_bid_below_limit_{i}_{h}"
                    )

        for i in range(self.cluster_size):  # Day-ahead scenarios
            for j in range(self.cluster_size):  # Intraday scenarios
                for h in range(24):  # Hours
                    if self.id_li[i, j, h] <= self.limit_price:
                        mdl.add_constraint(
                            ID_bids[i, j, h] == 0, f"ID_no_bid_below_limit_{i}_{j}_{h}"
                        )

    def create_model(self):
        mdl = Model(name=f"DayAheadOptimization_{self.date}")

        # Decision variables
        DA_bids = mdl.continuous_var_matrix(self.cluster_size, 24, name="DA_bids", lb=0)
        ID_bids = mdl.continuous_var_cube(self.cluster_size, self.cluster_size, 24, name="ID_bids", lb=0)
        eta = mdl.continuous_var(name="eta", lb=0)
        s = mdl.continuous_var_list(self.cluster_size, name="s", lb=0)

        # Objective function components
        DA_revenue = [mdl.sum(self.da[i, h] * DA_bids[i, h] for h in range(24)) for i in range(self.cluster_size)]
        ID_revenue = [
            mdl.sum(
                mdl.sum(self.id_li[i, j, h] * ID_bids[i, j, h] * self.id_probab[i, j] for h in range(24)) for j in range(self.cluster_size)
            )
            for i in range(self.cluster_size)
        ]
        total_revenue = [DA_revenue[i] + ID_revenue[i] for i in range(self.cluster_size)]
        total_cost = [
            mdl.sum(self.limit_price * DA_bids[i, h] for h in range(24)) +
            mdl.sum(self.limit_price * ID_bids[i, j, h] * self.id_probab[i, j] for j in range(self.cluster_size) for h in range(24))
            for i in range(self.cluster_size)
        ]
        contribution_margin = [total_revenue[i] - total_cost[i] for i in range(self.cluster_size)]
        
        # Expected contribution margin and CVaR term
        expected_contribution_margin = mdl.sum(self.da_probab[i] * contribution_margin[i] for i in range(self.cluster_size))
        cvar_term = mdl.sum(
            self.da_probab[i] * self.id_probab[i, j] * s[i] for i in range(self.cluster_size) for j in range(self.cluster_size)
        )
        cvar = eta - (1 / (1 - self.alpha_cvar)) * cvar_term

        # Maximize the objective
        mdl.maximize((1 - self.lambda_risk_aversion) * expected_contribution_margin + self.lambda_risk_aversion * cvar)

        # Add constraints using modularized methods
        self.add_forecast_constraints(mdl, DA_bids, ID_bids)
        self.add_min_da_constraints(mdl, 0.5, DA_bids)
        self.add_cvar_constraints(mdl, contribution_margin, eta, s)
        self.add_non_anticipativity_constraints(mdl, DA_bids, ID_bids)
        self.add_bidding_limit_constraints(mdl, DA_bids, ID_bids)

        self.mdl = mdl
        self.DA_bids = DA_bids
        self.ID_bids = ID_bids
        self.eta = eta

    def solve(self):
        # Set CPLEX parameters
        self.mdl.parameters.timelimit = 300
        self.mdl.parameters.mip.tolerances.mipgap = 0.01
        self.mdl.parameters.threads = 4
        self.mdl.parameters.preprocessing.presolve = 1

        # Solve the model
        solution = self.mdl.solve(log_output=True)
        if solution:
            print(f"Optimization successful for {self.date}!")
            self.DA_bids_optimal = np.array([
                [self.DA_bids[i, h].solution_value for h in range(24)] for i in range(self.cluster_size)
            ])
            self.ID_bids_optimal = np.array([
                [[self.ID_bids[i, j, h].solution_value for h in range(24)] for j in range(self.cluster_size)] for i in range(self.cluster_size)
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