import sys
sys.path.append('../..')

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use("grayscale")

# Load Gurobi license information
gurobi_dict = {}
with open('gurobi.lic', 'r') as file:
    for line in file:
        if line.strip() and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            gurobi_dict[key] = value

# Create an environment with Gurobi WLS license
params = {
    "CSMANAGER": gurobi_dict.get('CSMANAGER', ''),
    "CSAPIACCESSID": gurobi_dict.get('CSAPIACCESSID', ''),
    "CSAPISECRET": gurobi_dict.get('CSAPISECRET', ''), 
    "CSAPPNAME": gurobi_dict.get('CSAPPNAME', '')
}
env = gp.Env(params=params)
print("Gurobi environment parameters:", params)


class BiddingOptimization:
    """
    A class to optimize day-ahead (DA) and intraday (ID) bidding strategies for electricity markets.

    Parameters:
        limit_price (float): Price limit below which no bids will be placed (default 0).
        lambda_risk_aversion (float): Weight for the CVaR in the objective function (default 0.5).
        alpha_cvar (float): Confidence level for CVaR calculation (default 0.95).
        cluster_size (int): Number of clusters/scenarios (default 5).
        sell_at_limit (float): Percentage of forecasted generation committed at the price limit (default 0.25).
        penalty_weight (float): Weight for penalizing deviations from ideal bids (default 0.1).
        min_forecast_threshold (float): Minimum fraction of forecasted generation for constraints (default 0.9).
        max_forecast_threshold (float): Maximum fraction of forecasted generation for constraints (default 1.1).
        min_dayahead_threshold (float): Minimum fraction of forecasted generation for DA bids (default 0.5).
        max_dayahead_threshold (float): Maximum fraction of forecasted generation for DA bids (default 1).
    """
    def __init__(self, 
                 limit_price=0, 
                 lambda_risk_aversion=0.5, 
                 alpha_cvar=0.95, 
                 cluster_size=5, 
                 sell_at_limit=0.25, 
                 penalty_weight=0.1,
                 min_forecast_threshold=0.9,
                 max_forecast_threshold=1,
                 min_dayahead_threshold=0.5,
                 max_dayahead_threshold=1):
        
        # Initialize class attributes
        self.limit_price = limit_price
        self.lambda_risk_aversion = lambda_risk_aversion
        self.alpha_cvar = alpha_cvar
        self.cluster_size = cluster_size
        self.sell_at_limit = sell_at_limit
        self.penalty_weight = penalty_weight
        self.min_forecast_threshold = min_forecast_threshold
        self.max_forecast_threshold = max_forecast_threshold
        self.min_dayahead_threshold = min_dayahead_threshold
        self.max_dayahead_threshold = max_dayahead_threshold

    @staticmethod
    def load_csv(file_path):
        """
        Load a CSV file and convert it to a NumPy array, excluding the first column.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            np.ndarray: Data as a NumPy array.
        """
        data = pd.read_csv(file_path).iloc[:, 1:]
        return data.to_numpy()

    def load_data(self, date):
        """
        Load all necessary data for a given date.

        Parameters:
            date (str): The date for which data is loaded ('YYYY-MM-DD').
        """
        self.date = date

        # Load day-ahead data and probabilities
        self.da = self.load_csv(f"reduced_scenarios/C{self.cluster_size}/da/scenario_{date}.csv")
        self.da_probab = np.loadtxt(f"reduced_scenarios/C{self.cluster_size}/da/probability_{date}.txt")

        # Load intraday data and probabilities
        self.id_li = np.array([
            self.load_csv(f"reduced_scenarios/C{self.cluster_size}/id/scenario_{i}_{date}.csv")
            for i in range(self.cluster_size)
        ])
        self.id_probab = np.array([
            np.loadtxt(f"reduced_scenarios/C{self.cluster_size}/id/probability_{i}_{date}.txt")
            for i in range(self.cluster_size)
        ])

        # Normalize probabilities to sum to 1
        self.da_probab /= np.sum(self.da_probab)

        # Load forecasted generation data
        self.forecasted_generation = (1 - self.sell_at_limit) * pd.read_csv(
            "data/HKZ_forecast_da.csv", index_col=0
        ).loc[f"{date} 00:00:00":f"{date} 23:30:00", :]['frozen_fc_da'].values

    def prepare_schedule(self, date, results, da_prices_path, mcp_da_path, mcp_id_path, forecast_path, actual_path):
        """
        Prepares a schedule DataFrame for a given date by combining bid data, MCP data, 
        and forecasted/actual production.

        Parameters:
            date (str): The date for which the schedule is prepared ('YYYY-MM-DD').
            results (dict): Dictionary containing the results of DA bids for each date.
            da_prices_path (str): Path pattern for day-ahead prices CSV files with placeholders for date.
            mcp_da_path (str): Path pattern for day-ahead MCP CSV files with placeholders for date.
            mcp_id_path (str): Path pattern for intraday MCP CSV files with placeholders for date.
            forecast_path (str): Path to the forecasted generation CSV file.
            actual_path (str): Path to the actual production CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the schedule for the specified date.
        """
        # Prepare data for bids and prices
        data = []
        da_prices = pd.read_csv(da_prices_path.format(date=date)).iloc[:, 1:].to_numpy()
        for scenario in range(self.cluster_size):
            for hour in range(24):
                data.append({
                    "Scenario": scenario + 1,
                    "Hour": hour,
                    "DA_Bid": results[date]['DA_bids'][scenario, hour],
                    "DA_Price": da_prices[scenario, hour]
                })
        df = pd.DataFrame(data)

        # Load MCP data
        mcp_da = pd.read_csv(mcp_da_path.format(date=date))
        mcp_id = pd.read_csv(mcp_id_path.format(date=date))
        mcp_da['Hour'] = pd.to_datetime(mcp_da['delivery_begin']).dt.hour
        mcp_da.rename(columns={'DA_price': 'MCP_DA'}, inplace=True)
        mcp_id['Hour'] = pd.to_datetime(mcp_id['delivery_begin']).dt.hour
        mcp_id.rename(columns={'ID3_price': 'MCP_ID'}, inplace=True)

        # Merge DataFrames and filter by MCP
        merged = pd.merge(pd.merge(df, mcp_da, on='Hour', how='inner'), mcp_id, on='Hour', how='inner')
        filtered = merged[merged['DA_Price'] <= merged['MCP_DA']]
        aggregated = filtered.groupby('Hour', as_index=False).agg(
            DA_Bid=('DA_Bid', 'sum'),
            MCP_DA=('MCP_DA', 'first'),
            MCP_ID=('MCP_ID', 'first')
        )

        # Create schedule DataFrame and fill missing values
        schedule = pd.DataFrame({'Hour': range(24)}).merge(aggregated, on='Hour', how='left')
        schedule.fillna({'DA_Bid': 0, 'MCP_DA': 0, 'MCP_ID': 0}, inplace=True)

        # Add forecasted generation and actual production
        forecasted_gen = pd.read_csv(forecast_path, index_col=0).loc[
            f"{date} 00:00:00":f"{date} 23:30:00", :]['frozen_fc_da'].values
        actual_prod = pd.read_csv(actual_path, index_col=0).loc[
            f"{date} 00:00:00":f"{date} 23:30:00", :]['rt_signal'].values

        # Populate schedule with additional information
        schedule['forecast_gen'] = forecasted_gen
        schedule['DA_Bid'] = schedule.apply(
            lambda row: row['DA_Bid'] + self.sell_at_limit * row['forecast_gen'] if -2 < row['MCP_DA'] else row['DA_Bid'],
            axis=1
        )
        schedule['actual_prod'] = actual_prod
        schedule['volume_risk'] = schedule['actual_prod'] - schedule['DA_Bid']
        schedule['market_volume_risk'] = schedule['actual_prod'] - schedule['forecast_gen']

        return schedule
    
    def plot_schedule(self, schedule, date, path=None, plot=False):
        """
        Plots the schedule data including volume risks, market volume risks, and MCPs
        with a similar color scheme and styling as the bid accumulation plot.
    
        Parameters:
            schedule (pd.DataFrame): The schedule DataFrame containing bid and MCP data.
            date (str): The date of the schedule ('YYYY-MM-DD').
            path (str, optional): Path to save the plot. If None, the plot is not saved.
            plot (bool, optional): Whether to display the plot. Default is False.
        """
        fig, ax1 = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("white")  
        ax1.set_facecolor("white")  # No background color inside the plot
        
        bar_width = 0.4
        hours = schedule['Hour']
    
        # Define color map for volume risk bars
        cmap = plt.cm.get_cmap("RdYlGn")  # Red-Yellow-Green colormap
        norm = mcolors.Normalize(vmin=min(schedule['volume_risk']), vmax=max(schedule['volume_risk']))
        colors = [cmap(norm(v)) for v in schedule['volume_risk']]
    
        # Transparent bar charts for volume risks
        ax1.bar(hours - bar_width / 2, schedule['volume_risk'], alpha=0.5, label='Volume Risk (MWh)', 
                width=bar_width, color=colors, edgecolor='black')
        ax1.bar(hours + bar_width / 2, schedule['market_volume_risk'], alpha=0.5, label='Market Volume Risk (MWh)', 
                width=bar_width, color='gray', edgecolor='black')
    
        # Plot DA commitments with open triangle styling
        ax1.scatter(hours, schedule['DA_Bid'], label='DA Commitment (MWh)', edgecolor='green', 
                    facecolor='none', marker='^', s=100, linewidths=1.5)
    
        # Forecasted generation as dashed red line
        ax1.plot(hours, schedule['forecast_gen'], linestyle='--', color='red', 
                 label='Forecasted Generation (MW)', linewidth=2)
    
        # Configure primary y-axis
        # ax1.set_xlabel(f"Hours of {date}", fontsize=10)
        ax1.set_ylabel('Volume (MW)', fontsize=10)
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
        # MCPs on a secondary y-axis (matching bid plot style)
        ax2 = ax1.twinx()
        ax2.plot(hours, schedule['MCP_DA'], marker='o', linestyle='-', label='DA (€)', 
                 color='grey', markersize=7)
        ax2.plot(hours, schedule['MCP_ID'], marker='o', linestyle='-', label='ID (€)', 
                 color='darkgrey', markersize=7)
        ax2.set_ylabel('MCP [EUR/MWh]', fontsize=10)
    
        # Title with gray background to match previous plot
        plt.title(f"Risk Aversion: {self.lambda_risk_aversion}, CVaR: {self.alpha_cvar}", 
                  fontsize=12, pad=15, bbox=dict(facecolor="lightgray", edgecolor="none", pad=5))
    
        # Grid improvements
        ax1.grid(alpha=0.3)
        ax1.set_xticks(hours)
        ax1.set_xticklabels(hours, rotation=90, ha='right', fontsize=10)  # Rotate x-ticks
    
        # Adjust layout to give space for legend outside
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space on the right
    
        # Take legends completely outside the plot (center right)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.11, 1), fontsize=8, frameon=False)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.11, 0.70), fontsize=8, frameon=False)
    
        # Save or display the plot
        if path:
            plt.savefig(f'{path}_{self.lambda_risk_aversion}_{self.alpha_cvar}.png', bbox_inches='tight', dpi=300)
        if plot:
            plt.show()
        plt.close()
        
    def plot_accumulated_bids(self, DA_bids_optimal, da_prices, da_probab, date, alpha, cvar, path=None, plot=True):
        """
        Plots accumulated bids with bid price on the y-axis, volume reflected by open triangle size,
        and scenario leaf probability affecting the boundary color.
        """
    
        # Define hour labels (1-24)
        hours = np.arange(1, DA_bids_optimal.shape[1] + 1)
    
        # Flatten arrays for efficient plotting
        hour_values = np.tile(hours, DA_bids_optimal.shape[0])  
        bid_prices = da_prices.flatten()  
        bid_volumes = np.abs(DA_bids_optimal.flatten())  
        scenario_probs = np.repeat(da_probab, DA_bids_optimal.shape[1])  
    
        # Normalize bid volumes for marker size scaling
        min_vol, max_vol = np.min(bid_volumes), np.max(bid_volumes)
        sizes = 10 + (np.power(bid_volumes, 0.8) - np.power(min_vol, 0.8)) / (np.power(max_vol, 0.8) - np.power(min_vol, 0.8)) * 290  
    
        # Color normalization based on scenario probability
        cmap = plt.cm.get_cmap("RdYlGn")  
        norm = mcolors.Normalize(vmin=min(da_probab), vmax=max(da_probab))
        colors = [cmap(norm(prob)) for prob in scenario_probs]  
    
        fig, ax = plt.subplots(figsize=(5, 5))  
    
        fig.patch.set_facecolor("white")  
        ax.set_facecolor("white")  # No background color inside the plot
    
        # Scatter plot with open triangles (only the boundary is colored)
        scatter = ax.scatter(hour_values, bid_prices, edgecolors=colors, facecolors='none', marker='^', s=sizes, alpha=0.9, linewidths=1.5)
    
        # Labels and Titles
        ax.set_xticks(hours)
        ax.set_xticklabels(hours, rotation=90, ha='right', fontsize=10)  
        ax.set_xlabel("Hour", fontsize=10, labelpad=10)  
        ax.set_ylabel("Day-Ahead Bid Price [EUR/MWh]", fontsize=10)
    
        # Title with gray background but no spillover
        ax.set_title(f"α: {alpha} CVaR: {cvar}", fontsize=12, pad=15, 
                     bbox=dict(facecolor="lightgray", edgecolor="none", pad=5))
    
        # Grid and Formatting
        ax.grid(True, alpha=0.2)
    
        # Adjust layout to fit legend externally
        fig.subplots_adjust(right=0.85, bottom=0.15)  
    
        # Create a legend area for Scenario Probability & Volume
        legend_ax = fig.add_axes([0.90, 0.15, 0.02, 0.4])  
    
        # Colorbar for Scenario Leaf Probability
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical", cax=legend_ax, fraction=0.05, pad=0.05)
        cbar.set_label("Scenario Leaf Probability", fontsize=10)
        
        # Volume Legend
        volume_ax = fig.add_axes([0.87, 0.05, 0.02, 0.15])  
        legend_sizes = [10, 100, 300]  
        legend_labels = ["Small", "Medium", "Large"]
        for size, label in zip(legend_sizes, legend_labels):
            volume_ax.scatter([], [], edgecolor='black', facecolors='none', marker='^', s=size, label=label, linewidths=1.5)
        volume_ax.legend(title="Bid Volume", loc='center left', bbox_to_anchor=(0,5), fontsize=10, frameon=False)
        volume_ax.set_xticks([])
        volume_ax.set_yticks([])
        volume_ax.axis("off")
    
        # Save or Show Plot
        if path:
            plt.savefig(f'{path}/bids_plot_{date}_{alpha}_{cvar}.png', bbox_inches='tight', dpi=300)
        if plot:
            plt.show()
        plt.close()


    def add_zero_forecast_constraints(self, model, DA_bids, ID_bids):
        """
        Adds constraints to ensure no bids are placed when the forecasted generation is zero.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
            DA_bids (gurobipy.Var): Decision variables for day-ahead bids.
            ID_bids (gurobipy.Var): Decision variables for intraday bids.
        """
        epsilon = 1e-6  # Tolerance for zero forecast
        for h in range(24):
            forecast = self.forecasted_generation[h]
            if forecast <= epsilon:
                for i in range(self.cluster_size):
                    model.addConstr(
                        DA_bids[i, h] == 0,
                        name=f"total_DA_zero_bid_gen0_hour{h}"
                    )
                    for j in range(self.cluster_size):
                        model.addConstr(
                            ID_bids[i, j, h] == 0, 
                            name=f"ID_zero_bid_gen0_{i}_{j}_{h}")

    def add_forecast_constraints(self, model, min_threshold, max_threshold, DA_bids, ID_bids):
        """
        Adds constraints to ensure that the day-ahead (DA) and intraday (ID) bids for each hour 
        respect the forecasted generation limits and do not exceed actual production.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
            min_threshold (float): Minimum fraction of forecasted generation that must be committed to bids.
            max_threshold (float): Maximum fraction of forecasted generation that can be committed to bids.
            DA_bids (gurobipy.Var): Decision variables for day-ahead bids.
            ID_bids (gurobipy.Var): Decision variables for intraday bids.
        """
        epsilon = 1e-6  # Tolerance for zero forecast
        for h in range(24):  # Iterate over hours
            if self.forecasted_generation[h] > epsilon:
                # Minimum commitment constraint
                model.addConstr(
                    gp.quicksum(
                        DA_bids[i, h] for i in range(self.cluster_size)) +
                    gp.quicksum(
                        ID_bids[i, j, h] for i in range(self.cluster_size) for j in range(self.cluster_size)) 
                    >= min_threshold * self.forecasted_generation[h],
                    name=f"min_forecast_constraint_hour{h}"
                )
                # Maximum commitment constraint
                model.addConstr(
                    gp.quicksum(
                        DA_bids[i, h] for i in range(self.cluster_size)) +
                    gp.quicksum(
                        ID_bids[i, j, h] for i in range(self.cluster_size) for j in range(self.cluster_size)) 
                    <= max_threshold * self.forecasted_generation[h],
                    name=f"max_forecast_constraint_hour{h}"
                )

    def add_da_threshold_constraints(self, model, min_threshold, max_threshold, DA_bids):
        """
        Adds constraints for day-ahead (DA) bids to enforce minimum and maximum thresholds.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
            min_threshold (float): Minimum fraction of forecasted generation for DA bids.
            max_threshold (float): Maximum fraction of forecasted generation for DA bids.
            DA_bids (gurobipy.Var): Decision variables for day-ahead bids.
        """
        epsilon = 1e-6  # Tolerance for zero forecast
        for h in range(24):  # Iterate over hours
            if self.forecasted_generation[h] > epsilon:
                # Minimum DA bid threshold
                if min(self.da[:, h]) > self.limit_price:
                    model.addConstr(
                        gp.quicksum(
                            DA_bids[i, h] for i in range(self.cluster_size)) 
                        >= min_threshold * self.forecasted_generation[h],
                        name=f"min_da_constraint_hour{h}"
                    )
                # Maximum DA bid threshold
                model.addConstr(
                    gp.quicksum(
                        DA_bids[i, h] for i in range(self.cluster_size)) 
                    <= max_threshold * self.forecasted_generation[h],
                    name=f"max_da_constraint_hour{h}"
                )

    def add_bidding_limit_constraints(self, model, DA_bids, ID_bids):
        """
        Adds constraints to ensure no bids are placed below the limit price.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
            DA_bids (gurobipy.Var): Decision variables for day-ahead bids.
            ID_bids (gurobipy.Var): Decision variables for intraday bids.
        """
        for h in range(24):  # Iterate over hours
            epsilon = 1e-6  # Tolerance for zero forecast
            if self.forecasted_generation[h] > epsilon:
                for i in range(self.cluster_size):
                    # Ensure DA bids are zero if the price is below the limit
                    if self.da[i, h] <= self.limit_price:
                        model.addConstr(
                            DA_bids[i, h] == 0, 
                            name=f"DA_zero_bid_limit_{i}_{h}"
                        )
                    for j in range(self.cluster_size):
                        # Ensure ID bids are zero if the price is below the limit
                        if self.id_li[i, j, h] <= self.limit_price:
                            model.addConstr(
                                ID_bids[i, j, h] <= 0, 
                                name=f"ID_no_positive_sell_{i}_{j}_{h}"
                            )

    def add_cvar_constraints(self, model, contribution_margin, eta, s):
        """
        Adds constraints to enforce the Conditional Value at Risk (CVaR) calculation.
    
        Parameters:
            model (gurobipy.Model): The Gurobi model object.
            contribution_margin (2D list): Contribution margins for (DA, ID) scenarios.
            eta (gurobipy.Var): Gurobi variable representing the CVaR threshold.
            s (gurobipy.Var): Gurobi variables representing shortfall variables for each (DA, ID) scenario.
        """
        for i in range(self.cluster_size):
            for j in range(self.cluster_size):
                # Shortfall variable constraint
                model.addConstr(
                    s[i, j] >= eta - contribution_margin[i],
                    name=f"cvar_shortfall_{i}_{j}"
                )
                model.addConstr(s[i, j] >= 0, name=f"cvar_nonnegative_{i}_{j}")

    def create_model(self):
        """
        Creates the Gurobi optimization model, sets up decision variables, the objective function,
        and all necessary constraints.
        """
        # Initialize the model with the Gurobi environment
        model = gp.Model(env=env)

        # Decision Variables
        DA_bids = model.addVars(
            range(self.cluster_size), 24,
            lb=0,
            ub={(i, h): 1.1 * self.forecasted_generation[h] for i in range(self.cluster_size) for h in range(24)},
            name="DA_bids"
        )
        ID_bids = model.addVars(
            range(self.cluster_size), range(self.cluster_size), 24,
            lb={(i, j, h): -1.1 * self.forecasted_generation[h] for i in range(self.cluster_size) for j in range(self.cluster_size) for h in range(24)},
            ub={(i, j, h): 1.1 * self.forecasted_generation[h] for i in range(self.cluster_size) for j in range(self.cluster_size) for h in range(24)},
            name="ID_bids"
        )
        eta = model.addVar(lb=0, name="eta")  # CVaR threshold
        s = model.addVars(
            range(self.cluster_size), range(self.cluster_size), 
            lb=0, name="s"
        )  # Shortfall variables

        
        # Objective Function Components
        DA_revenue = [gp.quicksum(self.da[i, h] * DA_bids[i, h] for h in range(24)) for i in range(self.cluster_size)]
        ID_revenue = [
            gp.quicksum(
                gp.quicksum(self.id_li[i, j, h] * ID_bids[i, j, h] * self.id_probab[i, j] for h in range(24)) 
                for j in range(self.cluster_size)
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

        # Expected Contribution Margin and CVaR Term
        expected_contribution_margin = gp.quicksum(self.da_probab[i] * contribution_margin[i] for i in range(self.cluster_size))
        cvar_term = gp.quicksum(
            self.da_probab[i] * self.id_probab[i, j] * s[i, j] 
            for i in range(self.cluster_size) 
            for j in range(self.cluster_size)
        ) / (1 - self.alpha_cvar)  
        
        cvar = eta - cvar_term
        
        # Calculate mean day-ahead price for each hour
        mean_da_price = [gp.quicksum(self.da[i, h] for i in range(self.cluster_size)) / self.cluster_size for h in range(24)]

        # Penalty for intraday prices being very close to day-ahead prices
        id_price_deviation_penalty = gp.quicksum(
            (self.id_li[i, j, h] - mean_da_price[h])**2 * ID_bids[i, j, h]
            for i in range(self.cluster_size)
            for j in range(self.cluster_size)
            for h in range(24)
        )

        # Penalize DA bids far from the limit price
        DA_penalty = gp.quicksum((DA_bids[i, h] - self.limit_price)**2 for i in range(self.cluster_size) for h in range(24))
        # Penalize ID bids far from the limit price
        ID_penalty = gp.quicksum((ID_bids[i, j, h] - self.limit_price)**2 for i in range(self.cluster_size) for j in range(self.cluster_size) for h in range(24))

        # Objective Function
        # epsilon = 1e-6
        # model.setObjective(
        #     (1 - self.lambda_risk_aversion) * expected_contribution_margin +
        #     self.lambda_risk_aversion * cvar -
        #     epsilon * eta -
        #     self.penalty_weight * (DA_penalty + ID_penalty + id_price_deviation_penalty),
        #     GRB.MAXIMIZE
        # )

        model.setObjective(
            (1 - self.lambda_risk_aversion) * expected_contribution_margin +
            self.lambda_risk_aversion * cvar - 
            self.penalty_weight * (DA_penalty + ID_penalty),
            GRB.MAXIMIZE
        )

        # Add Constraints
        self.add_zero_forecast_constraints(model, DA_bids, ID_bids)
        self.add_bidding_limit_constraints(model, DA_bids, ID_bids)
        self.add_forecast_constraints(model, self.min_forecast_threshold, self.max_forecast_threshold, DA_bids, ID_bids)
        self.add_da_threshold_constraints(model, self.min_dayahead_threshold, self.max_dayahead_threshold, DA_bids)
        self.add_cvar_constraints(model, contribution_margin, eta, s)
        
        # Assign Model and Variables
        self.model = model
        self.DA_bids = DA_bids
        self.ID_bids = ID_bids
        self.eta = eta
        self.s = s
        
    def solve(self):
        """
        Solves the optimization model and prints the results or reports infeasibility.
        """
        # Set Gurobi parameters
        self.model.setParam('TimeLimit', 300)
        self.model.setParam('MIPGap', 0.01)
        self.model.setParam('Threads', 4)
        self.model.setParam('Presolve', 1)

        # Solve the model
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(f"Optimization successful for {self.date} (Risk Aversion: {self.lambda_risk_aversion}, CVaR: {self.alpha_cvar})!")
            
            self.DA_bids_optimal = np.array([
                [self.DA_bids[i, h].X for h in range(24)] for i in range(self.cluster_size)
            ])
            self.ID_bids_optimal = np.array([
                [[self.ID_bids[i, j, h].X for h in range(24)] for j in range(self.cluster_size)] for i in range(self.cluster_size)
            ])


            da_contribution_margin = sum(
                gp.quicksum(self.da[i, h] * self.DA_bids_optimal[i, h] for h in range(24)) -
                gp.quicksum(self.limit_price * self.DA_bids_optimal[i, h] for h in range(24))
                for i in range(self.cluster_size)
            ).getValue()
            
            id_contribution_margin = sum(
                gp.quicksum(
                    gp.quicksum(self.id_li[i, j, h] * self.ID_bids_optimal[i, j, h] * self.id_probab[i, j] for h in range(24))
                    for j in range(self.cluster_size)
                ) -
                gp.quicksum(
                    gp.quicksum(self.limit_price * self.ID_bids_optimal[i, j, h] * self.id_probab[i, j] for h in range(24))
                    for j in range(self.cluster_size)
                )
                for i in range(self.cluster_size)
            ).getValue()
            
            expected_contribution_margin = sum(
                self.da_probab[i] * (
                    gp.quicksum(self.da[i, h] * self.DA_bids_optimal[i, h] for h in range(24)) +
                    gp.quicksum(
                        gp.quicksum(self.id_li[i, j, h] * self.ID_bids_optimal[i, j, h] * self.id_probab[i, j] for h in range(24))
                        for j in range(self.cluster_size)
                    ) -
                    gp.quicksum(self.limit_price * self.DA_bids_optimal[i, h] for h in range(24)) -
                    gp.quicksum(
                        gp.quicksum(self.limit_price * self.ID_bids_optimal[i, j, h] * self.id_probab[i, j] for h in range(24))
                        for j in range(self.cluster_size)
                    )
                )
                for i in range(self.cluster_size)
            ).getValue()

            # Extract CVaR values
            cvar = self.eta.X - (
                sum(self.da_probab[i] * self.id_probab[i, j] * self.s[i, j].X
                    for i in range(self.cluster_size) 
                    for j in range(self.cluster_size)) / (1 - self.alpha_cvar)
            )
            # # Print results
            # print("DA Bids:", [
            #     [self.DA_bids[i, h].X for h in range(24)] for i in range(self.cluster_size)
            # ])
            # print("Objective Func:", self.model.getObjective().getValue())
            print("CVaR:", cvar)
            # print("da_cont_margin:", da_contribution_margin)
            # print("id_cont_margin:", id_contribution_margin)
            # print("expected_cont_margin:", expected_contribution_margin)

            # Store optimal solutions
            self.cvar = cvar
            self.da_contribution_margin = da_contribution_margin
            self.id_contribution_margin = id_contribution_margin
            self.expected_contribution_margin = expected_contribution_margin
            
        elif self.model.status == GRB.INFEASIBLE:
            print(f"Optimization failed for {self.date}: Model is infeasible")
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")
            print("The following constraints are causing infeasibility:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"Constraint {c.ConstrName} is in the IIS")
        else:
            print(f"Optimization failed for {self.date}: Status code {self.model.status}")

    def run(self, dates):
        """
        Runs the optimization process for multiple dates and stores the results.
    
        Parameters:
            dates (list): List of dates to run the optimization for.
    
        Returns:
            dict: A dictionary with optimization results for each date.
        """
        results = {}
        for date in dates:
            print(f"Running optimization for {date}...")
            try:
                # Load data and create model
                self.load_data(date)
                self.create_model()
                self.solve()
    
                if self.model.status == GRB.OPTIMAL:
                    print(np.sum(getattr(self, "DA_bids_optimal", None), axis=1))
                    print(self.da_probab)
                    results[date] = {
                        "DA_bids": getattr(self, "DA_bids_optimal", None),
                        "ID_bids": getattr(self, "ID_bids_optimal", None),
                        "da_contribution_margin": self.da_contribution_margin,
                        "id_contribution_margin": self.id_contribution_margin,
                        "expected_contribution_margin": self.expected_contribution_margin,
                        "cvar": self.cvar,
                    }
                else:
                    results[date] = {
                        "DA_bids": None,
                        "ID_bids": None,
                        "da_contribution_margin": None,
                        "id_contribution_margin": None,
                        "expected_contribution_margin": None,
                        "cvar": None,
                    }
            except Exception as e:
                print(f"Error occurred for {date}: {str(e)}")
                results[date] = {"error": str(e)}
        return results
    
