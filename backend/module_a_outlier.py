"""
ASTRA-Screen: Module A - Dynamic Outlier Detection System
Implements AEC-Q001 compliant Dynamic Part Average Testing (D-PAT)
combined with Multivariate Machine Learning (Robust Mahalanobis & Isolation Forest)
to flag latent defects escaping static datasheet limits.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest


class DynamicOutlierDetector:
    """
    Two-Tier Dynamic Outlier Detection:
    Tier 1: Statistical D-PAT (Part Average Testing, AEC-Q001 Standard).
    Tier 2: Multivariate ML Anomaly Ensemble (Isolation Forest + Mahalanobis Distance).
    """

    def __init__(
        self,
        dpat_sigma_multiplier: float = 3.0,
        contamination: float = 0.08,
        random_state: int = 42,
    ):
        self.dpat_mult = dpat_sigma_multiplier
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.lot_stats: Dict[str, Dict[str, float]] = {}
        self.mean_vector = None
        self.inv_cov_matrix = None
        self.feature_cols = ["Iddq_0h", "Ileak_0h", "PropDelay_0h"]

    def fit(self, df: pd.DataFrame) -> "DynamicOutlierDetector":
        """
        Learns the lot distribution statistics and fits multivariate anomaly models.
        """
        # Tier 1: Calculate Robust D-PAT limits per parameter (using Median and IQR)
        for col in self.feature_cols:
            median = df[col].median()
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = max(q75 - q25, 1e-6)
            # Pseudo-sigma = 0.7413 * IQR for normal distribution
            robust_sigma = 0.7413 * iqr

            lower_limit = median - (self.dpat_mult * robust_sigma)
            upper_limit = median + (self.dpat_mult * robust_sigma)

            self.lot_stats[col] = {
                "median": float(median),
                "q25": float(q25),
                "q75": float(q75),
                "robust_sigma": float(robust_sigma),
                "dpat_lower": float(lower_limit),
                "dpat_upper": float(upper_limit),
            }

        # Tier 2: Fit Multivariate Isolation Forest
        X = df[self.feature_cols].values
        self.iso_forest.fit(X)

        # Compute Robust Regularized Covariance via Ledoit-Wolf Shrinkage
        # Solves collinearity (e.g. Iddq vs Ileak) and strictly prevents singular matrix inversion
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        self.mean_vector = lw.location_
        self.inv_cov_matrix = np.linalg.pinv(lw.covariance_)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates components against learned dynamic limits and assigns anomaly scores.
        """
        result_df = df.copy()

        # Step 1: Compute D-PAT Violations
        dpat_flags = np.zeros(len(df), dtype=bool)
        dpat_deviations = []

        for col in self.feature_cols:
            stats = self.lot_stats[col]
            sigma = stats["robust_sigma"]
            median = stats["median"]
            upper = median + (self.dpat_mult * sigma)
            lower = median - (self.dpat_mult * sigma)

            # Keep lot_stats synced with current dynamic bounds
            stats["dpat_upper"] = float(upper)
            stats["dpat_lower"] = float(lower)

            col_dev = (df[col] - median) / (sigma + 1e-9)
            dpat_deviations.append(col_dev.values)

            # Check if value breaches dynamic limits
            col_flag = (df[col] > upper) | (df[col] < lower)
            dpat_flags = dpat_flags | col_flag.values

            result_df[f"{col}_DPAT_Upper"] = upper
            result_df[f"{col}_DPAT_ZScore"] = col_dev.round(2)

        result_df["DPAT_Flag"] = dpat_flags

        # Step 2: Multivariate ML Scoring
        X = df[self.feature_cols].values
        # Isolation Forest score: lower means more anomalous. Invert and scale to [0, 1]
        raw_iso_scores = self.iso_forest.score_samples(X)
        iso_min, iso_max = raw_iso_scores.min(), raw_iso_scores.max()
        iso_norm = 1.0 - ((raw_iso_scores - iso_min) / (iso_max - iso_min + 1e-6))
        result_df["IsoForest_Score"] = (iso_norm * 100).round(1)

        # Step 3: Mahalanobis Distance
        mahal_distances = []
        for x in X:
            dist = mahalanobis(x, self.mean_vector, self.inv_cov_matrix)
            mahal_distances.append(dist)
        mahal_distances = np.array(mahal_distances)
        result_df["Mahalanobis_Dist"] = mahal_distances.round(2)

        # Step 4: Unified Dynamic Risk Index (0 to 100)
        # Weighting: D-PAT violation provides strong baseline, supplemented by multivariate distance
        base_risk = iso_norm * 60.0
        mahal_risk = np.clip(mahal_distances / 5.0, 0, 1) * 40.0
        total_risk = base_risk + mahal_risk

        # If D-PAT threshold breached, ensure risk is at least 75
        total_risk = np.where(dpat_flags, np.maximum(total_risk, 78.0), total_risk)
        total_risk = np.clip(total_risk, 0.0, 100.0).round(1)

        result_df["Dynamic_Risk_Score"] = total_risk

        # Classification decision
        decisions = []
        for risk in total_risk:
            if risk >= 70.0:
                decisions.append("REJECT_DYNAMIC_OUTLIER")
            elif risk >= 45.0:
                decisions.append("WARNING_MARGINAL")
            else:
                decisions.append("PASS_FLIGHT_READY")

        result_df["Module_A_Decision"] = decisions
        return result_df

    def get_summary_metrics(self, evaluated_df: pd.DataFrame) -> Dict[str, any]:
        """
        Returns summary statistics for the lot screening report.
        """
        total = len(evaluated_df)
        rejected = (evaluated_df["Module_A_Decision"] == "REJECT_DYNAMIC_OUTLIER").sum()
        warning = (evaluated_df["Module_A_Decision"] == "WARNING_MARGINAL").sum()
        passed = (evaluated_df["Module_A_Decision"] == "PASS_FLIGHT_READY").sum()

        return {
            "total_screened": total,
            "passed_flight_ready": int(passed),
            "rejected_dynamic_outliers": int(rejected),
            "warning_marginal": int(warning),
            "lot_yield_percent": round((passed / total) * 100.0, 2),
            "lot_stats": self.lot_stats,
        }
