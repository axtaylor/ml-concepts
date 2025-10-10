import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
from dataclasses import dataclass, field

    
@dataclass
class LinearRegressionOLS:
    alpha: float = 0.05
    feature_names: list = field(default_factory=list)
    target: str = None
    
    X: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)
    
    theta: np.ndarray = field(default=None)
    coefficients: np.ndarray = field(default=None)
    intercept: float = None

    residuals: np.ndarray = field(default=None, repr=False)
    degrees_freedom: int = None
    rss: float = None
    tss: float = None
    ess: float = None
    mse: float = None
    rmse: float = None
    r_squared: float = None
    r_squared_2: float = None
    r_squared_adjusted: float = None
    log_likelihood: float = None
    aic: float = None
    bic: float = None

    variance_coefficient: np.ndarray = field(default=None, repr=False)
    std_error_coefficient: np.ndarray = field(default=None)
    t_stat_coefficient: np.ndarray = field(default=None)
    p_value_coefficient: np.ndarray = field(default=None)
    ci_low: np.ndarray = field(default=None)
    ci_high: np.ndarray = field(default=None)
    

    def __str__(self) -> str:
        if self.theta is None:
            return f"LinearRegressionOLS()"
        return (
            f"{' '*17}OLS Summary\n"
            f"{'='*45}\n"
            f"Target:                           {self.target}\n"
            f"Features:                         {list(self.feature_names)[1:]}\n"
            f"R²:                               {self.r_squared:.4f}\n"
            f"R² Adjusted:                      {self.r_squared_adjusted:.4f}\n"
            f"F-Statistic:                      {self.f_statistic:.4f}\n\n"
            f"MSE:                              {self.mse:.4f}\n"
            f"RMSE:                             {self.rmse:.4f}\n"
            f"RSS:                              {self.rss:.4f}\n"
            f"ESS:                              {self.ess:.4f}\n"
            f"TSS:                              {self.tss:.4f}\n\n"
            f"Residual Degrees of Freedom:      {self.degrees_freedom}\n"
            f"Observations:                     {self.X.shape[0]}\n"                    
            f"Features:                         {self.X.shape[1] - 1}\n\n"
            f"AIC:                              {self.aic:.4f}\n"     
            f"BIC:                              {self.bic:.4f}\n"  
            f"Log likelihood:                   {self.log_likelihood:.4f}\n"          
            f"{'='*45}\n"
            f"\n{' '*44}Model Weights"
            f"\n{'='*100}\n"
            f"{self.summary().to_string(index=False)}"
            f"\n{'='*100}\n"
        )
    

    def summary(self) -> pd.DataFrame:
        if self.theta is None:
            return pd.DataFrame()
        return pd.DataFrame(
        {
            "feature": feature,
            'coefficient': (np.round(coefficient,4) if abs(coefficient) > 0.0001 else np.format_float_scientific(coefficient, precision=2)),
            'se': (np.round(se,4) if abs(se) > 0.0001 else np.format_float_scientific(se, precision=2)),
            't_statistic': np.round(t, 4),
            'p_>_abs_t': f'{p:.3f}',
            f'conf_interval__{self.alpha}': [
                (np.round(low,3) if abs(low) > 0.0001 else np.format_float_scientific(low, precision=2)),
                (np.round(high,3) if abs(high) > 0.0001 else np.format_float_scientific(high, precision=2)),
            ],
        }
        for feature, coefficient, se, t, p, low, high in
        zip(self.feature_names, self.theta, self.std_error_coefficient, self.t_stat_coefficient, self.p_value_coefficient, self.ci_low, self.ci_high)
    )


    '''
    LinearRegressionOLS.fit(self, y, X, alpha = 0.05)

    Fit the linear regression model on the training data using the closed form ordinary least squares solution.
    
    '''
    def fit(self, y, X, alpha = 0.05):
        self.alpha = alpha
        self.feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {n}" for n in range(0, X.shape[1])]
        self.target = y.name if hasattr(y, 'name') else "Target"
        self.X, self.y = (np.asarray(X, dtype=float)), np.asarray(y, dtype=float)
        
        # Compute the coefficients using the normal equation
        xtx_inv = np.linalg.inv(self.X.T @ self.X)
        self.theta = xtx_inv @ (self.X.T @ self.y)                               

        # Assign coefficients and intercept
        self.intercept, self.coefficients = self.theta[0], self.theta[1:]
        
        # Predicted values and residuals
        y_hat = self.X @ self.theta        # or can call self.predict(self.X)
        y_bar = np.mean(self.y) 
        self.residuals = self.y - y_hat

        # Sum of squares
        self.rss = self.residuals @ self.residuals 
        self.tss = np.sum((self.y - y_bar)**2)                   
        self.ess = np.sum((y_hat - y_bar)**2)  

        # Residual degrees of freedom
        self.degrees_freedom = self.X.shape[0]-self.X.shape[1]

        # R squared
        self.r_squared = 1 - (self.rss / self.tss)
        self.r_squared_2 = self.ess / self.tss

        # Mean Squared Error and Root Mean Squared Error
        self.mse = self.rss / self.degrees_freedom 
        self.rmse = np.sqrt(self.mse)

        # F-statistic                          
        self.f_statistic = (self.ess / self.coefficients.shape[0]) / self.mse

        # Adjusted r2
        self.r_squared_adjusted = 1 - (1 - self.r_squared) * (self.X.shape[0] - 1) / self.degrees_freedom

        #self.aic = self.X.shape[0] * np.log(2 * np.pi * self.mse) + self.X.shape[0] + 2 * self.X.shape[1]
        #self.bic = self.X.shape[0] * np.log(2 * np.pi * self.mse) + self.X.shape[0] + np.log(self.X.shape[0]) * self.X.shape[1]

        self.log_likelihood = -self.X.shape[0]/2 * (np.log(2 * np.pi) + np.log(self.rss / self.X.shape[0]) + 1)
        self.aic = -2 * self.log_likelihood + 2 * self.X.shape[1]
        self.bic = -2 * self.log_likelihood + self.X.shape[1] * np.log(self.X.shape[0])

        # Variance of the coefficients (Covariance matrix)
        self.variance_coefficient = self.mse * xtx_inv     

        # Standard error of the coefficients 
        self.std_error_coefficient = np.sqrt(np.diag(self.variance_coefficient))

        # T statistic for the coefficients
        self.t_stat_coefficient = self.theta / self.std_error_coefficient

        # p > |T| for the coefficients
        self.p_value_coefficient = 2 * (1 - t_dist.cdf(abs(self.t_stat_coefficient), self.degrees_freedom))

        # T critical for the coefficients 
        t_crit = t_dist.ppf(1 - alpha/2, self.degrees_freedom)

        # Coefficient confidence intervals
        self.ci_low = self.theta - t_crit * self.std_error_coefficient
        self.ci_high = self.theta + t_crit * self.std_error_coefficient

        return self
    

    '''
    LinearRegressionOLS.predict(self, X)

    Make a simple prediction for y using a set of X values.
    
    '''
    def predict(self, X) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Error: Model is not fitted.")
        
        return np.asarray(X, dtype=float) @ self.coefficients + self.intercept
    


    '''========== Inference Functions =========='''



    '''
    LinearRegressionOLS.MarginalPredictions(self, X_test, alpha = 0.05)

    Create model predictions, return a dataframe of prediction statistics for each prediction

    '''
    def MarginalPredictions(self, X_test, alpha=0.05):
        if self.theta is None:
            raise ValueError("Error: Model is not fitted.")

        prediction_features = {j: f'{i.item():.2f}' for j, i in zip(self.feature_names[1:], X_test[0])}

        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test]) # Add intercept to test set
        prediction = X_test @ self.theta

        se_prediction = np.sqrt((X_test @ self.variance_coefficient @ X_test.T)).item()
        t_critical = t_dist.ppf(1 - alpha/2, self.degrees_freedom)

        ci_low, ci_high = (prediction - t_critical * se_prediction), (prediction + t_critical * se_prediction)

        t_stat = prediction / se_prediction
        p = 2 * (1 - t_dist.cdf(abs(t_stat), self.degrees_freedom))

        return pd.DataFrame({
            "x__input_vals": [prediction_features],
            "y__prediction": [np.round(prediction.item(), 4)],
            "se__prediction": [np.round(se_prediction,4)],
            "t_statistic__prediction": [np.round(t_stat.item(),4)],
            "p_>_abs_t__prediction": [p.item()],
            f"ci__prediction_{alpha}": [[np.round(ci_low.item(), 4), np.round(ci_high.item(), 4)]],
    })


    '''
    LinearRegressionOLS.HypothesisTesting(self, X_test, X_hyp, alpha=0.05)

    Test the effects of two predictions to determine statistical significance
    Test must be an array of (X) feature values.
    Hypothesis can either be a value for y as a float, or an array of (X) feature values.

    This is effectively the same as MarginalEffects, but with a hypothesis and report component.
    The functions are split for readability purposes.

    '''
    def HypothesisTesting(self, test, hyp, alpha=0.05):
        # Collecting feature names without constant term
        prediction_features = {j: f'{i.item():.2f}' for j, i in zip(self.feature_names[1:], test[0])}

        hypothesis_features = ({j: f'{i.item():.2f}' for j, i in zip(self.feature_names[1:], hyp[0])} if isinstance(hyp, np.ndarray) else {"Input Value (y)": f"{hyp}"})

        # Predict the two points (unless X_hyp is given as a float)
        prediction, hypothesis = self.predict(test), (self.predict(hyp) if isinstance(hyp, np.ndarray) else np.asarray(hyp))

        # Add a constant to the first prediction                                                                                  
        test = np.hstack([np.ones((test.shape[0], 1)), test])

        # Standard error of the first prediction     
        se = np.sqrt((test @ self.variance_coefficient @ test.T)).item()  

        # Confidence intervals for the first prediction
        t_critical = t_dist.ppf(1 - alpha/2, self.degrees_freedom)
        ci_low, ci_high = (prediction - t_critical * se),  (prediction + t_critical * se)                 

        # Determine the t statistic for the first prediction relative to the hypothesis
        t_stat = (prediction - hypothesis) / se          

        # P significance for the first prediction compared to the hypothesis
        p = 2 * (1 - t_dist.cdf(abs(t_stat), self.degrees_freedom))

        result = (
            f"\nFail to reject the null hypothesis: {prediction.item():.4f} is not statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
            f"\nConclude that outcome of {prediction_features}\ndoes not differ from {hypothesis_features}"
            if abs(t_stat.item()) < 1.96 else
            f"Reject the null hypothesis: {prediction.item():.4f} is statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
            f"Conclude that the outcomes of {prediction_features}\ndiffers significantly from {hypothesis_features}"
        )
        print(f"Marginal Effects Comparison:\n\nSignificance Analysis (p > |t|)\n1.96 > |{t_stat.item():.4f}| == {abs(t_stat.item()) < 1.96}\n", result)

        return pd.DataFrame({
            "x__prediction_vals": [prediction_features],
            "x__hypothesis_vals": [hypothesis_features],
            "y__prediction": [prediction.item()],
            "y__hypothesis": [hypothesis.item()],
            "se__prediction": [se],
            f"ci__prediction_{alpha}": [[np.round(ci_low.item(), 4), np.round(ci_high.item(), 4)]],
            "t_statistic__prediction_hypothesis": [t_stat.item()],
            "p_>_abs_t__prediction_hypothesis": [p.item()],
    }).T


    '''
    LinearRegressionOLS.VarianceInflationFactor(self)

    Perform a variance inflation factor check for multicollenearity in the feature set.

    
    '''
    def VarianceInflationFactor(self):
        if self.theta is None:
            raise ValueError("Error: Model is not fitted.")
        
        X = self.X[:,1:]
        n_features = X.shape[1]

        vif = []
        for i in range(n_features):
            # Constant bool column length of features, but this iteration is false
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False

            # X_j is the target, X_other are predictors
            X_j = X[:, i]
            X_other = X[:, mask]

            # Add intercept to X_other
            X_other_with_intercept = np.column_stack([np.ones(X_other.shape[0]), X_other])

            # Fit auxiliary regression: X_j ~ X_other
            theta_aux = np.linalg.inv(X_other_with_intercept.T @ X_other_with_intercept) @ (X_other_with_intercept.T @ X_j)
            y_hat_aux = X_other_with_intercept @ theta_aux

            # Calculate r-squared for auxiliary regression
            tss_aux = np.sum((X_j - np.mean(X_j))**2)
            rss_aux = np.sum((X_j - y_hat_aux)**2)
            r_squared_aux = 1 - (rss_aux / tss_aux)

            vif.append(1 / (1 - r_squared_aux) if r_squared_aux < 0.9999 else np.inf)

        return pd.DataFrame({
            'feature': self.feature_names[1:], 
            'VIF': np.round(vif, 4)
    })





'''========== Callable Functions =========='''



'''
Regression output that can compare multiple models.

'''
def stargazer(models, col_width=15) -> str:

    # Handle single model input
    if not isinstance(models, list):
        models = [models]

    # Verify models are fitted
    for i, model in enumerate(models):
        if model.theta is None:
            raise ValueError(f"Error: Model {i+1} is not fitted.")
    
    # Length of dividers by amount of models
    format_length = (
        38 if len(models) == 1 else
        52 if len(models) == 2  else
        68 if len(models) == 3  else
        81 if len(models) == 4  else
        100 if len(models) == 5  else
        110
    )
    header = (
        f"\n{"="*format_length}\n"
        "OLS Regression Results\n"
        f"{"="*format_length}\n"
        f"{'Dependent:':<20}" + "".join(f"{m.target:>{col_width}}" for m in models) + "\n"
        f"{"-"*format_length}\n"
    )

    # Add all features to the output, iteratively collecting new features in each model
    all_features = []
    for model in models:
        for feature in model.feature_names:
            if feature not in all_features:
                all_features.append(feature)
    
    # Add each row, iteratively.
    rows = []
    for feature in all_features:
        coef_row = f"{feature:<20}"
        se_row = " " * 20

        # Collect coefficients based on index of features
        for model in models:
            if feature in model.feature_names:
                feature_index = list(model.feature_names).index(feature)
                coef = model.theta[feature_index]
                se = model.std_error_coefficient[feature_index]
                p = model.p_value_coefficient[feature_index]
                
                stars = (
                    "***" if p < 0.01 else
                    "**" if p < 0.05 else
                    "*" if p < 0.1 else
                    ""
                )
                coef_fmt = f"{coef:.4f}{stars}" if abs(coef) > 0.0001 else f"{coef:.2e}{stars}"
                se_fmt = f"({se:.4f})" if abs(se) > 0.0001 else f"({se:.2e})"
                
                coef_row += f"{coef_fmt:>{col_width}}"
                se_row += f"{se_fmt:>{col_width}}"
            else:
                coef_row += " " * col_width
                se_row += " " * col_width

        # output the completed row for the feature from each model
        rows.append(" ")
        rows.append(coef_row)
        rows.append(se_row)

    # Collect model statistics and attribute names
    stats_lines = [
        ("R²", "r_squared"),
        ("Adjusted R²", "r_squared_adjusted"),
        ("F Statistic", "f_statistic"),
        ("Observations", lambda m: m.X.shape[0]),
        ("Log Likelihood", "log_likelihood"),
        ("AIC", "aic"),
        ("BIC", "bic")
    ]
    
    stats = f"{"-"*format_length}\n"

    # Call the attributes and append the results to the row
    for label, attr in stats_lines:
        stat_row = f"{label:<20}"
        for model in models:
            stat_row += f"{(attr(model) if callable(attr) else getattr(model, attr)):>{col_width}.3f}"
        stats += stat_row + "\n"
    
    # Return final constructed output
    return (
        header +
        "\n".join(rows) + "\n" +
        stats +
        f"{"="*format_length}\n"
        "Significance: *p<0.1; **p<0.05; ***p<0.01\n"
)