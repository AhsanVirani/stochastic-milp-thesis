import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.api import OLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")
pd.set_option('mode.use_inf_as_na', True)
plt.style.use("grayscale")

class Diagnostics:
    """
    Diagnostic tests and plots for linear regression model residuals and time series data.
    """
    
    def descriptive_statistics(self, df):
        """
        Generates and prints a descriptive statistics table for a DataFrame.
        
        Parameters:
        - df (pd.DataFrame): Data for generating descriptive statistics.
        
        Returns:
        - stats_df (pd.DataFrame): DataFrame containing descriptive statistics.
        """
        stats_df = df.describe().T
        stats_df['skew'] = df.skew()
        stats_df['kurtosis'] = df.kurtosis()
    
        stats_df.rename(columns={
            'mean': 'Mean',
            'std': 'Std Dev',
            'min': 'Min',
            '25%': '25th Percentile',
            '50%': 'Median',
            '75%': '75th Percentile',
            'max': 'Max',
            'skew': 'Skewness',
            'kurtosis': 'Kurtosis'
        }, inplace=True)
    
        stats_df = stats_df.round(3)
        print("Descriptive Statistics:\n")
        print(tabulate(stats_df, headers="keys", tablefmt="fancy_grid"))
        
        return stats_df

    def adf_test(self, series, alpha=0.05):
        """
        Performs the Augmented Dickey-Fuller test for stationarity.
        
        Parameters:
        - series (pd.Series): The time series data to test.
        - alpha (float): Significance level for the test.
        
        Returns:
        - dict: A dictionary with the ADF statistic, p-value, and stationarity verdict.
        """
        adf_stat, p_value, *_ = adfuller(series)
        result = {
            "ADF Statistic": adf_stat,
            "p-value": p_value,
            "Stationarity": "Stationary" if p_value < alpha else "Non-Stationary"
        }
        print(f"ADF Statistic: {result['ADF Statistic']}")
        print(f"p-value: {result['p-value']}")
        print(result["Stationarity"])
        
        return result


    def durbin_watson_test(self, residuals):
        """
        Performs the Durbin-Watson test for autocorrelation in residuals.
        
        Parameters:
        - residuals (array-like): Residual values.
        
        Returns:
        - dw_stat (float): Durbin-Watson statistic.
        """
        dw_stat = durbin_watson(residuals)
        print(f"Durbin-Watson statistic: {dw_stat}")
        print("No significant autocorrelation." if 1.5 <= dw_stat <= 2.5 else "Autocorrelation detected in residuals.")
        return dw_stat
    
    def breusch_pagan_test(self, residuals, exog):
        """
        Performs the Breusch-Pagan test for homoscedasticity in residuals.
        
        Parameters:
        - residuals (array-like): Residual values.
        - exog (pd.DataFrame or np.ndarray): Explanatory variables.
        
        Returns:
        - bp_stat (float): Breusch-Pagan test statistic.
        - bp_p_value (float): p-value of the test.
        """
        # Ensure `exog` has a constant for the test
        exog_with_constant = sm.add_constant(exog, has_constant='add')
        
        # Perform the Breusch-Pagan test
        bp_stat, bp_p_value, _, _ = het_breuschpagan(residuals, exog_with_constant)
        
        # Print results
        print(f"Breusch-Pagan test statistic: {bp_stat}, p-value: {bp_p_value}")
        if bp_p_value > 0.05:
            print("Homoscedasticity assumption holds.")
        else:
            print("Heteroscedasticity detected.")
        
        return bp_stat, bp_p_value

    def ljung_box_test(self, residuals, lags=24):
        """
        Performs the Ljung-Box test for autocorrelation in residuals.
        
        Parameters:
        - residuals (array-like): Residual values.
        - lags (int): Number of lags to test for autocorrelation.
        
        Returns:
        - lb_test (pd.DataFrame): Ljung-Box test result.
        """
        lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        print("Ljung-Box test p-values for autocorrelation:")
        print(lb_test)
        print("No significant autocorrelation in residuals." if lb_test['lb_pvalue'].iloc[0] > 0.05 else "Autocorrelation detected in residuals.")
        return lb_test
        
    def plot_histogram(self, residuals, title='Histogram of Residuals', path=None):
        """
        Plots a histogram of residuals for visual inspection of normality with a professional format.
        """
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")  # Ensure white background
        ax.hist(residuals, bins=20, color='dimgray', edgecolor='black', alpha=0.8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Residual', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if path:
            plt.savefig(path)
            
        plt.show()
        
    def check_multicollinearity(self, exog):
        """
        Checks for multicollinearity by calculating Variance Inflation Factors (VIF).
        
        Parameters:
        - exog (pd.DataFrame or np.ndarray): Explanatory variables.
        
        Returns:
        - vif_factors (pd.DataFrame): DataFrame containing VIF values.
        """
        if isinstance(exog, np.ndarray):
            exog = pd.DataFrame(exog)
        
        vif = [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])]
        vif_factors = pd.DataFrame({'VIF': vif, 'Variable': exog.columns})
        print("Variance Inflation Factors (VIF):")
        print(vif_factors)
        print("No significant multicollinearity detected." if vif_factors['VIF'].max() < 5 else "High multicollinearity detected in predictors.")
        return vif_factors
        
    def rolling_window_stability(self, model, window_size=50, path=None):
        """
        Analyzes model parameter stability using rolling window OLS regression.
        
        Parameters:
        - model: Fitted statsmodels OLS model.
        - window_size: Size of the rolling window.
        - path: If provided, saves the plot to the given file path.

        Returns:
        - stability (bool): True if model parameters are stable, otherwise False.
        """
        rolling_params = []
        for i in range(window_size, len(model.model.endog)):
            X_rolling = model.model.exog[i - window_size:i]
            y_rolling = model.model.endog[i - window_size:i]
            rolling_model = OLS(y_rolling, X_rolling).fit()
            rolling_params.append(rolling_model.params)
        
        rolling_params_df = pd.DataFrame(rolling_params, columns=model.model.exog_names)
        
        # Plot rolling coefficients
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        for param in rolling_params_df.columns:
            ax.plot(rolling_params_df[param], label=param, color='dimgray', alpha=0.8)
        
        ax.set_xlabel("Observations", fontsize=10)
        ax.set_ylabel("Coefficient Value", fontsize=10)
        ax.legend(fontsize=10, frameon=False, bbox_to_anchor=(1, 1))
        ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        
        if path:
            plt.savefig(path, bbox_inches='tight')
        plt.show()
        
        stability = all(rolling_params_df.std() < 0.05)
        print("Model parameters are stable over time." if stability else "Model parameters show signs of instability.")
        return stability

    def normality_test(self, residuals, alpha=0.05, path=None):
        """
        Tests the normality of residuals using Shapiro-Wilk and Anderson-Darling tests and plots a QQ plot.
        
        Parameters:
        - residuals: Residuals from the model.
        - alpha: Significance level for hypothesis tests.
        - path: If provided, saves the QQ plot to the given file path.

        Returns:
        - Dictionary containing test statistics and results.
        """
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        anderson_result = stats.anderson(residuals, dist='norm')
        
        print("Shapiro-Wilk Test:")
        print(f"Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
        print("Residuals likely normal." if shapiro_p > alpha else "Residuals not normal.")
        
        print("\nAnderson-Darling Test:")
        print(f"Statistic: {anderson_result.statistic:.4f}")
        for sl, cv in zip(anderson_result.significance_level, anderson_result.critical_values):
            print(f"Significance Level: {sl}%, Critical Value: {cv:.4f}")
        
        # QQ Plot
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, color='dimgray', alpha=0.6, label="Residuals")
        ax.plot(osm, slope * osm + intercept, color='black', linestyle='--', linewidth=1.5, label="Normal Line")
        
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        ax.set_ylabel("Sample Quantiles", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10, frameon=False, loc="best")
        
        if path:
            plt.savefig(path, bbox_inches='tight')
        plt.show()
        
        return {
            "Shapiro-Wilk": {"statistic": shapiro_stat, "p-value": shapiro_p},
            "Anderson-Darling": {
                "statistic": anderson_result.statistic,
                "critical_values": anderson_result.critical_values.tolist(),
                "significance_levels": anderson_result.significance_level.tolist()
            }
        }
    
    def plot_acf_pacf(self, series, lags=100, acf_title="Autocorrelation Function (ACF)", pacf_title="Partial Autocorrelation Function (PACF)", path=None):
        """
        Plots the Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions.
        
        Parameters:
        - series: Time series data.
        - lags: Number of lags to display.
        - acf_title: Title for the ACF plot.
        - pacf_title: Title for the PACF plot.
        - path: If provided, saves the plots to the given file path.

        Returns:
        - None
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        
        plot_acf(series, lags=lags, ax=axes[0])
        axes[0].set_title(acf_title)
        axes[0].set_ylabel("Autocorrelation")
        
        plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
        axes[1].set_title(pacf_title)
        
        plt.tight_layout()
        
        if path:
            plt.savefig(path, bbox_inches='tight')
        plt.show()
