from dataclasses import dataclass, field
from typing import Optional, List
import warnings
import numpy as np
from scipy.stats import t as t_dist

@dataclass
class LinearRegressionOLS:
    alpha: float = None
    feature_names: list = field(default_factory=list)
    target: str = None
    X: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)
    degrees_freedom: int = None
    xtx_inv: np.ndarray = field(default=None, repr=False)
    theta: np.ndarray = field(default=None)
    coefficients: np.ndarray = field(default=None)
    intercept: float = None
    residuals: np.ndarray = field(default=None, repr=False)
    rss: float = None
    tss: float = None
    ess: float = None
    mse: float = None
    rmse: float = None
    r_squared: float = None
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

    def _model_is_fitted(self):
        if self.theta is None:
            raise ValueError(
                "Model is not fitted. Call 'fit' with arguments "
                "before using this method."
            )

    def __str__(self):
        self._model_is_fitted()
        return summary(self)

    def coefficient_table(self):
        self._model_is_fitted()
        return [
        {
            "feature": feature,
            'coefficient': (np.round(coefficient,4) if abs(coefficient) > 0.0001 else np.format_float_scientific(coefficient, precision=2)),
            'se': (np.round(se,4) if abs(se) > 0.0001 else np.format_float_scientific(se, precision=2)),
            't_statistic': np.round(t, 4),
            'P>|t|': f'{p:.3f}',
            f'conf_interval__{self.alpha}': [
                (np.round(low,3) if abs(low) > 0.0001 else np.format_float_scientific(low, precision=2)),
                (np.round(high,3) if abs(high) > 0.0001 else np.format_float_scientific(high, precision=2)),
            ],
        }
        for feature, coefficient, se, t, p, low, high in
        zip(self.feature_names, self.theta, self.std_error_coefficient, self.t_stat_coefficient, self.p_value_coefficient, self.ci_low, self.ci_high)
    ]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        alpha: float = 0.05
        ) -> 'LinearRegressionOLS':

        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        X_array, y_array = (np.asarray(X, dtype=float)), np.asarray(y, dtype=float)

        if X_array.size == 0 or y_array.size == 0:
            raise ValueError("X and y cannot be empty")

        if len(X_array.shape) != 2:
            raise ValueError(f"X must be 2D, got shape {X_array.shape} instead.")

        if len(y_array.shape) != 1:
            if len(y_array.shape) == 2 and y_array.shape[1] == 1:
                y_array = y_array.flatten()
            else:
                raise ValueError(f"y must be 1D, got shape {y_array.shape} instead.")

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                f"X and y must have same number of observations. "
                f"Got X: {X_array.shape[0]}, y: {y_array.shape[0]} instead."
        )
        if X_array.shape[0] <= X_array.shape[1]:
            raise ValueError(
                f"Insufficient observations. Need n > k, "
                f"got n={X_array.shape[0]}, k={X_array.shape[1]} instead."
        )
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha} instead.")

        if np.any(~np.isfinite(X_array)):
            raise ValueError("X contains NaN or infinite values.")

        if np.any(~np.isfinite(y_array)):
            raise ValueError("y contains NaN or infinite values.")

        self.feature_names = (
            X.columns if hasattr(X, 'columns')
            else feature_names if feature_names is not None
            else ['const', *[f"Feature {i}" for i in range(1,X.shape[1])]] # Generic names for no args np array inputs, assuming const first.
        )
        self.target = (
            y.name if hasattr(y, 'name')
            else target_name if target_name is not None
            else "Dependent"
        )
        self.alpha = alpha
        self.X, self.y = X_array, y_array
        self.degrees_freedom = self.X.shape[0]-self.X.shape[1]

        # Cholesky decomposition fit
        xtx = self.X.T @ self.X
        try:
            L = np.linalg.cholesky(xtx)
            self.theta = np.linalg.solve(L.T, np.linalg.solve(L, self.X.T @ self.y))
            I = np.eye(xtx.shape[0])
            self.xtx_inv = np.linalg.solve(L.T, np.linalg.solve(L, I))
        except np.linalg.LinAlgError:
            raise ValueError(
            "Matrix X'X is not positive definite. This typically indicates:\n"
            "- Perfect multicollinearity between features\n"
            "- Insufficient observations (n < k)\n"
            "- Constant or duplicate columns in X"
            )
        cond = np.linalg.cond(xtx)
        if cond > 1e10:
            warnings.warn(
                f"X'X matrix is ill-conditioned (cond={cond:.2e}).\n"
                f"Results may be unreliable. Consider:\n"
                f"- Removing collinear features\n"
                f"- Scaling features\n"
                f"- Using regularization\n",
                UserWarning,
                stacklevel=2
        )
        # Predict
        self.intercept, self.coefficients = self.theta[0], self.theta[1:]
        y_hat = self.X @ self.theta #self.predict(self.X)
        y_bar = np.mean(self.y)
        self.residuals = self.y - y_hat

        # Squared Residuals
        self.rss = self.residuals @ self.residuals
        self.ess = np.sum((y_hat - y_bar)**2)
        self.tss = np.sum((self.y - y_bar)**2)

        # Loss
        self.mse = self.rss / self.degrees_freedom
        self.rmse = np.sqrt(self.mse)

        # Model
        self.f_statistic = (
            (self.ess / self.coefficients.shape[0]) / self.mse
            if self.coefficients.shape[0] > 0 and self.mse > 1e-15
            else np.inf
        )
        self.r_squared = 1 - (self.rss / self.tss)
        self.r_squared_adjusted = 1 - (1 - self.r_squared) * (self.X.shape[0] - 1) / self.degrees_freedom
        self.log_likelihood = -self.X.shape[0]/2 * (np.log(2 * np.pi) + np.log(self.rss / self.X.shape[0]) + 1)
        self.aic = -2 * self.log_likelihood + 2 * self.X.shape[1]
        self.bic = -2 * self.log_likelihood + self.X.shape[1] * np.log(self.X.shape[0])

        # Feature
        self.variance_coefficient = self.mse * self.xtx_inv  # Variance of the coefficients (Covariance matrix)
        self.std_error_coefficient = np.sqrt(np.diag(self.variance_coefficient))
        self.t_stat_coefficient = self.theta / self.std_error_coefficient
        self.p_value_coefficient = 2 * (1 - t_dist.cdf(abs(self.t_stat_coefficient), self.degrees_freedom))

        t_crit = t_dist.ppf(1 - alpha/2, self.degrees_freedom)
        self.ci_low = self.theta - t_crit * self.std_error_coefficient
        self.ci_high = self.theta + t_crit * self.std_error_coefficient

        return self

    def predict(self, X, alpha=0.05, return_table=False):
        self._model_is_fitted()
        if return_table == False:
            return (np.asarray(X, dtype=float) @ self.coefficients + self.intercept)

        prediction_features = {j: f'{i.item():.2f}' for j, i in zip(self.feature_names[1:], X[0])}
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        prediction = X @ self.theta
        se_prediction = np.sqrt((X @ self.variance_coefficient @ X.T)).item()
        t_critical = t_dist.ppf(1 - alpha/2, self.degrees_freedom)
        ci_low, ci_high = (prediction - t_critical * se_prediction), (prediction + t_critical * se_prediction)
        t_stat = prediction / se_prediction
        p = 2 * (1 - t_dist.cdf(abs(t_stat), self.degrees_freedom))

        return ({
            "features": [prediction_features],
            "prediction": [np.round(prediction.item(), 4)],
            "std_error": [np.round(se_prediction,4)],
            "t_statistic": [np.round(t_stat.item(),4)],
            "P>|t|": [p.item()],
            f"ci_low_{alpha}": [np.round(ci_low.item(), 4)],
            f"ci_high_{alpha}": [np.round(ci_high.item(), 4)],
    })

    def hypothesis_testing(self, test, hyp, alpha=0.05):
        self._model_is_fitted()
        critical = np.round(t_dist.ppf(1 - alpha/2, self.degrees_freedom),2)
        prediction_features = {j: f'{i.item():.2f}' for j, i in zip(self.feature_names[1:], test[0])}
        hypothesis_features = (
            {j: f'{i.item():.2f}' for j, i in zip(self.feature_names[1:], hyp[0])}
            if isinstance(hyp, np.ndarray)
            else {f"{self.target}": f"{hyp}"}
        )
        test = np.hstack([np.ones((test.shape[0], 1)), test])
        prediction, hypothesis = test @ self.theta, (
            np.hstack([np.ones((hyp.shape[0], 1)), hyp]) @ self.theta
            if isinstance(hyp, np.ndarray)
            else np.asarray(hyp)
        )
        se = np.sqrt((test @ self.variance_coefficient @ test.T)).item()
        t_stat = (prediction - hypothesis) / se
        p = 2 * (1 - t_dist.cdf(abs(t_stat), self.degrees_freedom))

        result = (
            f"Significance Analysis (p > |t|)\n{critical} > |{t_stat.item():.4f}| == {abs(t_stat.item()) < critical}\n"
            f"\nFail to reject the null hypothesis: {prediction.item():.4f} is not statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
            f"\nConclude that outcome of {prediction_features}\ndoes not differ from {hypothesis_features}"
            if abs(t_stat.item()) < critical else
            f"Significance Analysis (p > |t|)\n{critical} > |{t_stat.item():.4f}| == {abs(t_stat.item()) < critical}\n"
            f"\nReject the null hypothesis: {prediction.item():.4f} is statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
            f"\nConclude that the outcomes of {prediction_features}\ndiffers significantly from {hypothesis_features}"
        )
        return (
        {
            "summary": result,
            "table": {
                "feature_labels": [prediction_features],
                "hypothesis_labels": [hypothesis_features],
                "prediction": [prediction.item()],
                "hypothesis": [hypothesis.item()],
                "t-statistic": [t_stat.item()],
                "P>|t|": [p.item()],
            },
        }
    )

    def variance_inflation_factor(self):
        self._model_is_fitted()
        X = self.X[:,1:]
        n_features, vif = X.shape[1], []

        for i in range(n_features):
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            X_j = X[:, i]                                                                        # Target
            X_other_with_intercept = np.column_stack([np.ones(X[:, mask].shape[0]), X[:, mask]]) # Other Features

            # Auxiliary fit
            xtx = X_other_with_intercept.T @ X_other_with_intercept
            theta_aux = np.linalg.solve(xtx, X_other_with_intercept.T @ X_j)
            y_hat_aux = X_other_with_intercept @ theta_aux
            tss_aux = np.sum((X_j - np.mean(X_j))**2)
            if tss_aux < 1e-10:
                vif.append(np.inf)
                continue
            rss_aux = np.sum((X_j - y_hat_aux)**2)
            r_squared_aux = 1 - (rss_aux / tss_aux)
            vif.append(1 / (1 - r_squared_aux) if r_squared_aux < 0.9999 else np.inf)

        return ({
            'feature': self.feature_names[1:],
            'VIF': np.round(vif, 4)
    })

    def robust_se(self, type="HC3"):
        self._model_is_fitted()
        n, k = self.X.shape
        h = np.sum(self.X @ self.xtx_inv * self.X, axis=1)
        sr = self.residuals.reshape(-1, 1).flatten()**2
        HC_ = {
            "HC0": lambda sr, n_obs, k_regressors, leverage: sr,
            "HC1": lambda sr, n_obs, k_regressors, leverage: (n_obs / (n_obs - k_regressors)) * sr,
            "HC2": lambda sr, n_obs, k_regressors, leverage: sr / (1 - leverage),
            "HC3": lambda sr, n_obs, k_regressors, leverage: sr / ((1 - leverage) ** 2),
        }
        try:
            omega_diagonal = HC_[type](sr, n, k, h)
            X_omega = self.X * np.sqrt(omega_diagonal)[:, None]                   # Multiply each X row by X*(diagonal weights)^(0.5)
            robust_cov = self.xtx_inv @ (X_omega.T @ X_omega) @ self.xtx_inv      # Sandwich
            robust_se = np.sqrt(np.diag(robust_cov))                              # Diagonal extract the var-cov
            robust_t_stat = self.theta / robust_se

            return {
            "feature": self.feature_names,
            "robust_se": robust_se,
            "robust_t": robust_t_stat,
            "robust_p": 2 * (1 - t_dist.cdf(abs(robust_t_stat), self.degrees_freedom)),
            #"type": type,
            #"covariance": robust_cov,
        }
        except KeyError:
            raise ValueError("Select 'HC0', 'HC1', 'HC2', 'HC3'")


def summary(models, col_width=15, compression=20):
    if not isinstance(models, list):
        models = [models]

    for i, model in enumerate(models):
        if model.theta is None:
            raise ValueError(f"Error: Model {i+1} is not fitted.")

    format_length = compression + (len(models)*col_width)
    header = (
        f"\n{"="*format_length}\n"
        "OLS Regression Results\n"
        f"{"="*format_length}\n"
        f"{'Dependent:':<{compression}}" + "".join(f"{m.target:>{col_width}}" for m in models) + "\n"
        f"{"-"*format_length}\n"
    )
    all_features = []
    for model in models:
        for feature in model.feature_names:
            if feature not in all_features:
                all_features.append(feature)
    rows = []
    for feature in all_features:
        coef_row = f"{feature:<{compression}}"
        se_row = " " * compression
        #t_row = " " * compression
        for model in models:
            if feature in model.feature_names:
                feature_index = list(model.feature_names).index(feature)
                coef = model.theta[feature_index]
                se = model.std_error_coefficient[feature_index]
                p = model.p_value_coefficient[feature_index]
                #t = model.t_stat_coefficient[feature_index]
                stars = (
                    "***" if p < 0.01 else
                    "**" if p < 0.05 else
                    "*" if p < 0.1 else
                    ""
                )
                coef_fmt = (
                    f"{coef:.4f}{stars}"
                    if abs(coef) > 0.0001
                    else f"{coef:.2e}{stars}"
                )
                se_fmt = (
                    f"({se:.4f})"
                    if abs(se) > 0.0001
                    else f"({se:.2e})"
                )
                #t_fmt = f"{t:.4f}" if abs(t) > 0.0001 else f"({t:.2e})"
                coef_row += f"{coef_fmt:>{col_width}}"
                se_row += f"{se_fmt:>{col_width}}"
                #t_row += f"{t_fmt:>{col_width}}"
            else:
                coef_row += " " * col_width
                se_row += " " * col_width
                #t_row += " " * col_width
        rows.append(" ")
        rows.append(coef_row)
        rows.append(se_row)
        #rows.append(t_row)
    stats_lines = [
        ("R-squared", "r_squared"),
        ("Adjusted R-squared", "r_squared_adjusted"),
        ("F Statistic", "f_statistic"),
        ("Observations", lambda m: m.X.shape[0]),
        ("Log Likelihood", "log_likelihood"),
        ("AIC", "aic"),
        ("BIC", "bic")
    ]
    stats = f"\n{"-"*format_length}\n"
    for label, attr in stats_lines:
        stat_row = f"{label:<{compression}}"
        for model in models:
            stat_row += f"{(attr(model) if callable(attr) else getattr(model, attr)):>{col_width}.3f}"
        stats += stat_row + "\n"
    return (
        header +
        "\n".join(rows) + "\n" +
        stats +
        f"{"="*format_length}\n"
        "*p<0.1; **p<0.05; ***p<0.01\n"
)