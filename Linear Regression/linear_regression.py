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

    variance_coefficient: np.ndarray = field(default=None, repr=False)
    std_error_coefficient: np.ndarray = field(default=None)
    t_stat_coefficient: np.ndarray = field(default=None)
    p_value_coefficient: np.ndarray = field(default=None)
    ci_low: np.ndarray = field(default=None)
    ci_high: np.ndarray = field(default=None)
    
    def __str__(self) -> str:
        return (
            f"{' '*15}OLS Summary\n"
            f"{'='*40}\n"
            f"Target: {self.target}\n"
            f"R²: {self.r_squared:.4f}\n"
            f"F-Statistic: {self.f_statistic:.4f}\n\n"
            f"MSE: {self.mse:.4f}\n"
            f"RMSE: {self.rmse:.4f}\n"
            f"RSS: {self.rss:.4f}\n"
            f"ESS: {self.ess:.4f}\n"
            f"TSS: {self.tss:.4f}\n\n"
            f"Residual Degrees of Freedom: {self.degrees_freedom}\n"
            f"Observations : {self.X.shape[0]}\n"                        # (shape[0])
            f"Features : {self.X.shape[1] - 1}\n"                        # (shape[1]-1)
            f"{'='*40}\n"
            f"\n{' '*44}Model Weights"
            f"\n{'='*100}\n"
            f"{self.coefficient_dataframe().to_string(index=False)}"
            f"\n{'='*100}\n"
        )
    
    #def __repr__(self) -> str:
    #    return f"{self}"
    
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
    
    def predict(self, X_test) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Error: Model is not fitted.")
        return np.asarray(X_test, dtype=float) @ self.coefficients + self.intercept
    
    def coefficient_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
        {
            "feature_name": f,
            'coefficient': np.round(c,4),
            'std_error_coefficient': np.round(s, 4),
            't_statistic': np.round(t, 4),
            'p_>_abs_t': np.round(p, 4),
            f'conf_interval__{self.alpha}': [np.round(low,4), np.round(high,4)],
        }
    for f, c, s, t, p, low, high in zip(self.feature_names, self.theta, self.std_error_coefficient, self.t_stat_coefficient, self.p_value_coefficient, self.ci_low, self.ci_high)
    )


# Marginal effects comparing a prediction to a hypothesis
def marginal_effects(X_test, X_hyp, model, alpha=0.05):

    # Collecting features without constant term
    prediction_features = {j: f'{i.item():.2f}' for j, i in zip(model.feature_names[1:], X_test[0])}
    hypothesis_features = {j: f'{i.item():.2f}' for j, i in zip(model.feature_names[1:], X_hyp[0])}

    # Predict the two points
    prediction, hypothesis = model.predict(X_test), model.predict(X_hyp)    

    # Add a constant to the first prediction                                                                                  
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    # Matrix multiply the first prediction to the models coefficient variance matrix and the first prediction transpose 
    prediction_variance = X_test @ model.variance_coefficient @ X_test.T

    # Standard error of the first prediction     
    se = np.sqrt(prediction_variance.item())     

    # Confidence intervals for the first prediction
    t_critical = t_dist.ppf(1 - alpha/2, model.degrees_freedom)
    ci_low, ci_high = prediction - t_critical * se,  prediction + t_critical * se                 

    # Determine the t statistic for the first prediction relative to the hypothesis
    t_stat = (prediction - hypothesis) / se          

    # P significance for the first prediction compared to the hypothesis
    p = 2 * (1 - t_dist.cdf(abs(t_stat), model.degrees_freedom))

    result = (
        f"\nAccept the null hypothesis: {prediction.item():.4f} is not statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
        f"Conclude that outcome of {prediction_features}\ndoes not differ from {hypothesis_features}"
        if abs(t_stat.item()) < 1.96 else
        f"Reject the null hypothesis: {prediction.item():.4f} is statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
        f"Conclude that the outcomes of {prediction_features}\ndiffers significantly from {hypothesis_features}"
    )
    print(f"Marginal Effects Comparison:\n\nSignificance Analysis (p > |t|)\n1.96 > |{t_stat.item():.4f}| == {abs(t_stat.item()) < 1.96}\n", result)

    marginal_effects = pd.DataFrame({
        "x__prediction_vals": [prediction_features],
        "x__hypothesis_vals": [hypothesis_features],
        "y__prediction": [prediction.item()],
        "y__hypothesis": [hypothesis.item()],
        "se__prediction": [se],
        f"ci__prediction_{alpha}": [[np.round(ci_low.item(), 4), np.round(ci_high.item(), 4)]],
        "t_statistic__prediction_hypothesis": [t_stat.item()],
        "p_>_abs_t__prediction_hypothesis": [p.item()],
    }).T

    marginal_effects.columns = ['Marginal Effects - prediction vs. hypothesis']
    return marginal_effects




# The Marginal effect for specific prediction
def model_prediction(X_test, model, alpha=0.05):

    prediction_features ={j: f'{i.item():.2f}' for j, i in zip(model.feature_names[1:], X_test[0])}
    prediction = model.predict(X_test)
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    prediction_variance = X_test @ model.variance_coefficient @ X_test.T
    se = np.sqrt(prediction_variance.item())

    t_critical = t_dist.ppf(1 - alpha/2, model.degrees_freedom)
    ci_low = prediction - t_critical * se
    ci_high = prediction + t_critical * se

    t_stat = prediction / se
    p = 2 * (1 - t_dist.cdf(abs(t_stat), model.degrees_freedom))

    return pd.DataFrame({
        "x__input_vals": [prediction_features],
        "y__prediction": [np.round(prediction.item(), 4)],
        "se__prediction": [np.round(se,4)],
        "t_statistic__prediction": [np.round(t_stat.item(),4)],
        "p_>_abs_t__prediction": [np.round(p.item(),4)],
        f"ci__prediction_{alpha}": [[np.round(ci_low.item(), 4), np.round(ci_high.item(), 4)]],
    })