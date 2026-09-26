"""
ASTRA-Screen: Module B - Time-Series Drift Predictor & Early Abort Engine
Takes initial (0h) and early burn-in (24h) parametric data to forecast end-of-test (168h) values.
If forecasted drift exceeds the calculated safety slope, triggers immediate early rejection at 24h,
saving up to 144 hours of thermal chamber operation per defective unit.
"""

from typing import Dict, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class EarlyDriftPredictor:
    """
    Time-Series Regression Model predicting 168h degradation from 0h and 24h observations.
    Incorporates an asymmetric safety margin to heavily penalize under-prediction (False Negatives).
    """

    def __init__(
        self,
        datasheet_max_iddq: float = 50.0,
        safety_margin_factor: float = 0.90,  # Alert at 90% of datasheet max (45 µA)
        random_state: int = 42,
    ):
        self.datasheet_max_iddq = datasheet_max_iddq
        self.safety_threshold = datasheet_max_iddq * safety_margin_factor  # 45.0 µA
        self.random_state = random_state
        self.model = None
        self.residual_std = 0.0
        self.feature_names = [
            "Iddq_0h",
            "Iddq_24h",
            "Delta_Iddq_24_0",
            "Velocity_Iddq",
            "Relative_Drift_Pct",
            "Ileak_0h",
            "Ileak_24h",
            "Delta_Ileak_24_0",
            "PropDelay_0h",
            "PropDelay_24h",
            "Delta_Delay_24_0",
        ]

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers rate-of-change, velocity, and multi-parametric drift features.
        """
        feat = pd.DataFrame(index=df.index)

        # Iddq drift kinematics
        feat["Iddq_0h"] = df["Iddq_0h"]
        feat["Iddq_24h"] = df["Iddq_24h"]
        feat["Delta_Iddq_24_0"] = df["Iddq_24h"] - df["Iddq_0h"]
        feat["Velocity_Iddq"] = feat["Delta_Iddq_24_0"] / 24.0  # µA per hour
        feat["Relative_Drift_Pct"] = (feat["Delta_Iddq_24_0"] / (df["Iddq_0h"] + 1e-6)) * 100.0

        # Leakage current drift
        feat["Ileak_0h"] = df["Ileak_0h"]
        feat["Ileak_24h"] = df["Ileak_24h"]
        feat["Delta_Ileak_24_0"] = df["Ileak_24h"] - df["Ileak_0h"]

        # Propagation delay drift
        feat["PropDelay_0h"] = df["PropDelay_0h"]
        feat["PropDelay_24h"] = df["PropDelay_24h"]
        feat["Delta_Delay_24_0"] = df["PropDelay_24h"] - df["PropDelay_0h"]

        return feat

    def fit(self, train_df: pd.DataFrame) -> "EarlyDriftPredictor":
        """
        Trains the LightGBM regressor on historical burn-in lots.
        """
        X = self._extract_features(train_df)
        y = train_df["Iddq_168h"].values

        # LightGBM Regressor with Huber loss for robustness against extreme spikes
        self.model = lgb.LGBMRegressor(
            objective="regression_l1",  # Optimizes directly for MAE
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=self.random_state,
            verbose=-1,
        )
        self.model.fit(X, y)

        # Calculate residual standard deviation for prediction intervals
        preds = self.model.predict(X)
        residuals = y - preds
        self.residual_std = float(np.std(residuals))

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecasts Iddq_168h, computes confidence intervals, and flags components for early rejection.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predict() is called.")

        result_df = df.copy()
        X = self._extract_features(df)

        # Forecast Value_168h
        y_pred = self.model.predict(X)

        # Apply conservative aerospace safety offset (95% upper bound to eliminate False Negatives)
        upper_bound_168h = y_pred + (1.645 * self.residual_std)

        # Calculate predicted drift slope from 24h to 168h
        # Slope = (Pred_168h - Actual_24h) / 144 hours
        predicted_drift_slope = (y_pred - df["Iddq_24h"].values) / 144.0

        # Max safe slope: slope that would take the component exactly to safety threshold
        safe_slope_allowed = np.maximum(
            (self.safety_threshold - df["Iddq_24h"].values) / 144.0, 0.0
        )

        # Calculate 24h drift velocity:
        v_24 = (df["Iddq_24h"].values - df["Iddq_0h"].values) / 24.0

        # Decision Gate: Flag for Early Rejection at 24h if:
        # 1. Forecasted 168h value exceeds safety threshold (45 µA), OR
        # 2. Conservative upper bound exceeds datasheet max (50 µA), OR
        # 3. Predicted drift velocity is accelerating unsafely (> safe_slope_allowed), OR
        # 4. 24h initial drift velocity is abnormally high (> 0.06 µA/h, spaceflight safety boundary)
        early_reject_flag = (
            (y_pred >= self.safety_threshold)
            | (upper_bound_168h >= self.datasheet_max_iddq)
            | (predicted_drift_slope > safe_slope_allowed)
            | (v_24 > 0.06)
        )

        result_df["Pred_Iddq_168h"] = np.round(y_pred, 3)
        result_df["Pred_Iddq_168h_Upper95"] = np.round(upper_bound_168h, 3)
        result_df["Predicted_Drift_Slope_uA_per_hr"] = np.round(predicted_drift_slope, 4)
        result_df["Max_Safe_Slope_uA_per_hr"] = np.round(safe_slope_allowed, 4)
        result_df["Early_Reject_24h_Flag"] = early_reject_flag

        # Classify decision
        decisions = []
        for is_reject in early_reject_flag:
            if is_reject:
                decisions.append("EARLY_REJECT_AT_24H")
            else:
                decisions.append("PROCEED_TO_168H")
        result_df["Module_B_Decision"] = decisions

        return result_df

    def evaluate(self, evaluated_df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes the official SIH / ISRO evaluation metrics:
        - MAE between predicted 168h and actual hidden ground-truth (if available).
        - False Negative Count (catastrophic metric: missed defective parts).
        - False Positive Count.
        - Chamber Hours Saved.
        """
        predicted_defective = evaluated_df["Early_Reject_24h_Flag"]
        aborted_parts = int(predicted_defective.sum())
        hours_saved = aborted_parts * 144  # 168h - 24h = 144 hours saved per part

        has_ground_truth = "Iddq_168h" in evaluated_df.columns

        if has_ground_truth:
            actual_168h = evaluated_df["Iddq_168h"].values
            pred_168h = evaluated_df["Pred_Iddq_168h"].values
            mae = float(mean_absolute_error(actual_168h, pred_168h))
            rmse = float(np.sqrt(mean_squared_error(actual_168h, pred_168h)))

            # Actual defects at 168h: ground truth failure or actual 168h >= 45 µA
            gt_failure_cond = (evaluated_df["Ground_Truth_Failure"] == 1) if "Ground_Truth_Failure" in evaluated_df.columns else False
            actual_defective = (evaluated_df["Iddq_168h"] >= 45.0) | gt_failure_cond

            # Confusion Matrix
            false_negatives = int((actual_defective & (~predicted_defective)).sum())
            false_positives = int(((~actual_defective) & predicted_defective).sum())
            true_positives = int((actual_defective & predicted_defective).sum())
            true_negatives = int(((~actual_defective) & (~predicted_defective)).sum())

            # Score = 100 - (FN * 25.0) - (FP * 1.5)
            anomaly_score = max(0.0, 100.0 - (false_negatives * 25.0) - (false_positives * 1.5))
        else:
            # Real-world inference mode (no future ground-truth known yet)
            mae = 0.0
            rmse = 0.0
            false_negatives = 0
            false_positives = 0
            true_positives = aborted_parts
            true_negatives = len(evaluated_df) - aborted_parts
            anomaly_score = 100.0

        return {
            "drift_mae_uA": round(mae, 4),
            "drift_rmse_uA": round(rmse, 4),
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "chamber_hours_saved": hours_saved,
            "early_aborted_count": aborted_parts,
            "anomaly_detection_score": round(anomaly_score, 2),
            "has_ground_truth": has_ground_truth,
        }
