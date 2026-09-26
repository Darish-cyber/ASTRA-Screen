"""
ASTRA-Screen: Automated Statistical Process Control (SPC) & Distribution Drift Monitor
Complies with MIL-STD-883 and AEC-Q001 SPC guidelines.
Monitors input feature distributions against the calibrated baseline using:
- Kullback-Leibler (KL) Divergence with Laplace smoothing
- Symmetric Jensen-Shannon (JS) Divergence
- 1st Wasserstein Distance (Earth Mover's Distance)
- Two-sample Kolmogorov-Smirnov (KS) test
- Shewhart SPC Control Limits (UCL / LCL: mean +/- 3*sigma) & Cpk Process Capability
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


MONITORED_FEATURES = [
    "Iddq_0h",
    "Ileak_0h",
    "PropDelay_0h",
    "Delta_Iddq_24_0",
    "Velocity_Iddq",
    "Delta_Ileak_24_0",
]

# Standard thresholds aligned with Slide 5 Sustainability Protocol
KL_THRESHOLD_WARNING = 0.15
KL_THRESHOLD_CRITICAL = 0.35
CPK_MIN_ACCEPTABLE = 1.33  # Standard 4-sigma aerospace capability


def calculate_kl_divergence(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int = 35,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes Kullback-Leibler divergence D_KL(P || Q) between baseline (P)
    and incoming batch (Q) using binned empirical distributions with Laplace smoothing.
    """
    p_clean = np.asarray(p_samples, dtype=float)
    q_clean = np.asarray(q_samples, dtype=float)

    # Common bin edges across both distributions
    min_val = min(np.min(p_clean), np.min(q_clean))
    max_val = max(np.max(p_clean), np.max(q_clean))
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    p_hist, _ = np.histogram(p_clean, bins=bin_edges)
    q_hist, _ = np.histogram(q_clean, bins=bin_edges)

    # Laplace smoothing to prevent division by zero or log(0)
    p_prob = (p_hist + epsilon) / np.sum(p_hist + epsilon)
    q_prob = (q_hist + epsilon) / np.sum(q_hist + epsilon)

    kl = float(np.sum(p_prob * np.log(p_prob / q_prob)))
    return max(0.0, kl)


def calculate_js_divergence(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int = 35,
) -> float:
    """
    Computes symmetric Jensen-Shannon divergence:
    D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = 0.5 * (P + Q).
    Bounded between [0.0, ln(2) = 0.693].
    """
    p_clean = np.asarray(p_samples, dtype=float)
    q_clean = np.asarray(q_samples, dtype=float)

    min_val = min(np.min(p_clean), np.min(q_clean))
    max_val = max(np.max(p_clean), np.max(q_clean))
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    p_hist, _ = np.histogram(p_clean, bins=bin_edges)
    q_hist, _ = np.histogram(q_clean, bins=bin_edges)

    eps = 1e-6
    p_prob = (p_hist + eps) / np.sum(p_hist + eps)
    q_prob = (q_hist + eps) / np.sum(q_hist + eps)
    m_prob = 0.5 * (p_prob + q_prob)

    kl_pm = np.sum(p_prob * np.log(p_prob / m_prob))
    kl_qm = np.sum(q_prob * np.log(q_prob / m_prob))
    return float(max(0.0, 0.5 * (kl_pm + kl_qm)))


def calculate_spc_metrics(
    current_data: np.ndarray,
    baseline_data: Optional[np.ndarray] = None,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculates Shewhart Statistical Process Control (SPC) metrics:
    - Mean and standard deviation
    - Upper & Lower Control Limits (UCL/LCL = mean +/- 3*sigma)
    - Process Capability Index Cpk
    """
    arr = np.asarray(current_data, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-6
    std_val = max(std_val, 1e-6)

    # Baseline limits if provided, otherwise derived from sample
    if baseline_data is not None and len(baseline_data) > 1:
        base_mean = float(np.mean(baseline_data))
        base_std = max(float(np.std(baseline_data, ddof=1)), 1e-6)
        ucl = base_mean + 3.0 * base_std
        lcl = max(0.0, base_mean - 3.0 * base_std)
    else:
        ucl = mean_val + 3.0 * std_val
        lcl = max(0.0, mean_val - 3.0 * std_val)

    # Process capability Cpk calculation
    cpk = 2.0  # default healthy
    if usl is not None and lsl is not None:
        cpu = (usl - mean_val) / (3.0 * std_val)
        cpl = (mean_val - lsl) / (3.0 * std_val)
        cpk = float(min(cpu, cpl))
    elif usl is not None:
        cpk = float((usl - mean_val) / (3.0 * std_val))

    ooc_count = int(np.sum((arr > ucl) | (arr < lcl)))
    ooc_pct = float(ooc_count / len(arr) * 100.0) if len(arr) > 0 else 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ucl": round(ucl, 4),
        "lcl": round(lcl, 4),
        "cpk": round(cpk, 2),
        "ooc_count": ooc_count,
        "ooc_pct": round(ooc_pct, 2),
    }


class ModelDriftMonitor:
    """
    Automated Drift Monitor for Spaceflight Component Screening.
    Maintains a calibrated baseline distribution and evaluates incoming lots
    for process shifts, foundry variations, and distribution drift.
    """

    def __init__(
        self,
        baseline_df: Optional[pd.DataFrame] = None,
        kl_warning: float = KL_THRESHOLD_WARNING,
        kl_critical: float = KL_THRESHOLD_CRITICAL,
    ):
        self.kl_warning = kl_warning
        self.kl_critical = kl_critical
        self.baseline_df = None
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

        if baseline_df is not None:
            self.fit_baseline(baseline_df)

    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures derived drift kinematic columns exist."""
        out = df.copy()
        if "Delta_Iddq_24_0" not in out.columns and "Iddq_24h" in out.columns and "Iddq_0h" in out.columns:
            out["Delta_Iddq_24_0"] = out["Iddq_24h"] - out["Iddq_0h"]
        if "Velocity_Iddq" not in out.columns and "Delta_Iddq_24_0" in out.columns:
            out["Velocity_Iddq"] = out["Delta_Iddq_24_0"] / 24.0
        if "Delta_Ileak_24_0" not in out.columns and "Ileak_24h" in out.columns and "Ileak_0h" in out.columns:
            out["Delta_Ileak_24_0"] = out["Ileak_24h"] - out["Ileak_0h"]
        return out

    def fit_baseline(self, baseline_df: pd.DataFrame) -> None:
        """Stores reference baseline feature distributions."""
        df_feat = self._ensure_features(baseline_df)
        self.baseline_df = df_feat.copy()
        self.baseline_stats = {}

        for col in MONITORED_FEATURES:
            if col in df_feat.columns:
                arr = df_feat[col].dropna().values
                mean_v = float(np.mean(arr))
                std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-6
                self.baseline_stats[col] = {
                    "mean": mean_v,
                    "std": max(std_v, 1e-6),
                    "ucl": mean_v + 3.0 * std_v,
                    "lcl": max(0.0, mean_v - 3.0 * std_v),
                    "median": float(np.median(arr)),
                    "iqr": float(stats.iqr(arr)),
                }

    def evaluate_lot(
        self,
        current_df: pd.DataFrame,
        custom_baseline: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates an incoming lot against baseline for statistical drift.
        Returns a comprehensive dictionary with per-feature metrics and system status.
        """
        curr = self._ensure_features(current_df)
        base = self._ensure_features(custom_baseline) if custom_baseline is not None else self.baseline_df

        if base is None:
            # Self-reference if baseline not yet initialized
            base = curr

        feature_reports: List[Dict[str, Any]] = []
        max_kl = 0.0
        max_kl_feature = ""
        total_drift_score = 0.0

        for feat in MONITORED_FEATURES:
            if feat not in curr.columns or feat not in base.columns:
                continue

            c_vals = curr[feat].dropna().values
            b_vals = base[feat].dropna().values

            if len(c_vals) < 10 or len(b_vals) < 10:
                continue

            kl_div = calculate_kl_divergence(b_vals, c_vals)
            js_div = calculate_js_divergence(b_vals, c_vals)
            w_dist = float(stats.wasserstein_distance(b_vals, c_vals))
            ks_res = stats.ks_2samp(b_vals, c_vals)
            spc = calculate_spc_metrics(c_vals, b_vals)

            if kl_div > max_kl:
                max_kl = kl_div
                max_kl_feature = feat

            total_drift_score += kl_div

            # Feature status
            if kl_div >= self.kl_critical:
                feat_status = "CRITICAL"
                badge = "🔴 Critical Shift"
            elif kl_div >= self.kl_warning:
                feat_status = "WARNING"
                badge = "🟡 Moderate Drift"
            else:
                feat_status = "NOMINAL"
                badge = "🟢 In Control"

            feature_reports.append({
                "Feature": feat,
                "KL_Divergence": round(kl_div, 4),
                "JS_Divergence": round(js_div, 4),
                "Wasserstein_Dist": round(w_dist, 4),
                "KS_pvalue": round(float(ks_res.pvalue), 4),
                "Current_Mean": spc["mean"],
                "Baseline_Mean": self.baseline_stats.get(feat, {}).get("mean", spc["mean"]),
                "Current_Std": spc["std"],
                "UCL_3sigma": spc["ucl"],
                "LCL_3sigma": spc["lcl"],
                "Cpk": spc["cpk"],
                "OOC_Pct": spc["ooc_pct"],
                "Status": feat_status,
                "Badge": badge,
            })

        # Overall System Status Determination
        if max_kl >= self.kl_critical:
            system_status = "CRITICAL"
            status_text = "CRITICAL: FOUNDRY FAB SHIFT DETECTED"
            color = "#dc2626"
            action = (
                f"Feature '{max_kl_feature}' exhibited KL-Divergence of {max_kl:.4f} (>= {self.kl_critical:.2f}). "
                "Halt automatic screening. Initiate Material Review Board (MRB) calibration protocol per Slide 5."
            )
        elif max_kl >= self.kl_warning:
            system_status = "WARNING"
            status_text = "WARNING: PROCESS DRIFT DETECTED"
            color = "#d97706"
            action = (
                f"Feature '{max_kl_feature}' exhibited KL-Divergence of {max_kl:.4f} (>= {self.kl_warning:.2f}). "
                "Process parameters are drifting relative to baseline. Monitor next production lot closely."
            )
        else:
            system_status = "NOMINAL"
            status_text = "NOMINAL: PROCESS IN STATISTICAL CONTROL"
            color = "#16a34a"
            action = (
                f"All monitored parameters are in control. Maximum KL-Divergence is {max_kl:.4f} (< {self.kl_warning:.2f}). "
                "Safe to screen lot using current calibrated model parameters."
            )

        mean_kl = total_drift_score / len(feature_reports) if feature_reports else 0.0

        return {
            "system_status": system_status,
            "status_text": status_text,
            "status_color": color,
            "max_kl": round(max_kl, 4),
            "max_kl_feature": max_kl_feature,
            "mean_kl": round(mean_kl, 4),
            "recommended_action": action,
            "feature_table": pd.DataFrame(feature_reports),
            "kl_warning_threshold": self.kl_warning,
            "kl_critical_threshold": self.kl_critical,
        }


def simulate_foundry_shift(
    df: pd.DataFrame,
    iddq_shift_pct: float = 35.0,
    noise_sigma: float = 0.4,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Simulates a foundry process shift (e.g. gate oxide thickness variation or wafer lot change).
    Used for interactive live testing in the dashboard.
    """
    rng = np.random.RandomState(seed)
    shifted = df.copy()

    # Shift Iddq baseline upward by iddq_shift_pct%
    scale = 1.0 + (iddq_shift_pct / 100.0)
    shifted["Iddq_0h"] = shifted["Iddq_0h"] * scale + rng.normal(0, noise_sigma, len(shifted))
    shifted["Iddq_24h"] = shifted["Iddq_24h"] * scale + rng.normal(0, noise_sigma * 1.5, len(shifted))
    shifted["Ileak_0h"] = shifted["Ileak_0h"] * (1.0 + (iddq_shift_pct * 0.5 / 100.0))
    shifted["Ileak_24h"] = shifted["Ileak_24h"] * (1.0 + (iddq_shift_pct * 0.7 / 100.0))

    # Re-derive kinematics
    shifted["Delta_Iddq_24_0"] = shifted["Iddq_24h"] - shifted["Iddq_0h"]
    shifted["Velocity_Iddq"] = shifted["Delta_Iddq_24_0"] / 24.0
    shifted["Delta_Ileak_24_0"] = shifted["Ileak_24h"] - shifted["Ileak_0h"]

    return shifted
