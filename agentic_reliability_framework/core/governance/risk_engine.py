# agentic_reliability_framework/core/governance/risk_engine.py
"""
Bayesian Risk Scoring Engine – Conjugate priors (online) + HMC (offline).

This engine combines two complementary Bayesian approaches:
1. Conjugate Beta priors for each action category – fast online updates.
2. A Hamiltonian Monte Carlo logistic regression (using NUTS) that learns
   complex patterns (time of day, user role, environment) from historical data.

At runtime, the final risk score is a weighted average of the conjugate prior
mean and the HMC prediction, with weights determined by the amount of data
available for the specific context (so the HMC model contributes more when
its training data is sufficient).

The engine also provides an update_outcome() method for online learning,
and a train_hmc() method that can be called periodically (e.g., via cron)
to retrain the HMC model on accumulated incident data.
"""

import os
import json
import threading
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    ResourceType,
    PermissionLevel,
    Environment,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Action Categories (same as before)
# =============================================================================
class ActionCategory(str, Enum):
    DATABASE = "database"
    NETWORK = "network"
    COMPUTE = "compute"
    SECURITY = "security"
    DEFAULT = "default"


# =============================================================================
# Conjugate Beta Priors (Online Model)
# =============================================================================
PRIORS = {
    ActionCategory.DATABASE: (1.5, 8.0),   # Beta(α, β) – more pessimistic
    ActionCategory.NETWORK: (1.2, 10.0),
    ActionCategory.COMPUTE: (1.0, 12.0),
    ActionCategory.SECURITY: (2.0, 10.0),
    ActionCategory.DEFAULT: (1.0, 10.0),
}

class BetaStore:
    """Thread‑safe store of Beta posterior parameters per category."""
    def __init__(self):
        self._data: Dict[ActionCategory, Tuple[float, float]] = {
            cat: PRIORS[cat] for cat in ActionCategory
        }
        self._lock = threading.RLock()

    def get(self, category: ActionCategory) -> Tuple[float, float]:
        with self._lock:
            return self._data[category]

    def update(self, category: ActionCategory, success: bool):
        with self._lock:
            alpha, beta = self._data[category]
            if success:
                alpha += 1
            else:
                beta += 1
            self._data[category] = (alpha, beta)


# =============================================================================
# HMC Model (Offline Bayesian Logistic Regression)
# =============================================================================
class HMCModel:
    """
    Logistic regression with NUTS, trained on historical incidents.
    Features include:
        - action category (one‑hot encoded)
        - sin(hour), cos(hour) for cyclical time
        - environment (production vs. other)
        - user role (if available)
        - policy violation count
        - etc.
    The model is saved to disk after training and hot‑loaded at runtime.
    """

    def __init__(self, model_path: str = "hmc_model.json"):
        self.model_path = model_path
        self.coefficients: Optional[Dict[str, float]] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.is_ready = False
        self._load()

    def _load(self):
        """Load pre‑trained coefficients from JSON."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    data = json.load(f)
                self.coefficients = data.get("coefficients", {})
                self.feature_names = data.get("feature_names", [])
                scaler_params = data.get("scaler")
                if scaler_params:
                    self.feature_scaler = StandardScaler()
                    self.feature_scaler.mean_ = np.array(scaler_params["mean"])
                    self.feature_scaler.scale_ = np.array(scaler_params["scale"])
                self.is_ready = True
                logger.info(f"HMC model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load HMC model: {e}")

    def _save(self, trace, feature_names, scaler):
        """Save posterior means as coefficients to JSON."""
        coeffs = {}
        for var in trace.posterior.data_vars:
            if var.startswith("beta_") or var == "alpha":
                mean_val = float(trace.posterior[var].mean().values)
                coeffs[var] = mean_val
        data = {
            "coefficients": coeffs,
            "feature_names": feature_names,
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
        }
        with open(self.model_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"HMC model saved to {self.model_path}")

    def predict(self, intent: InfrastructureIntent, context: Dict[str, Any]) -> Optional[float]:
        """
        Return the HMC‑predicted probability of incident for the given intent.
        Returns None if model not ready.
        """
        if not self.is_ready or self.coefficients is None or self.feature_names is None:
            return None

        # Build feature vector in the same order as feature_names
        # We need a function that maps intent + context to a named feature dict.
        # For demonstration, we create a simple fixed‑order list based on feature_names.
        # In a production system, you would store the exact feature engineering pipeline.
        # Here we assume feature_names are like: ['sin_hour', 'cos_hour', 'env_prod', 'user_role', 'cat_database', ...]
        hour = datetime.datetime.now().hour
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)
        env_prod = 1.0 if hasattr(intent, "environment") and intent.environment == Environment.PROD else 0.0
        user_role = 0.0  # placeholder

        cat = self._categorize_intent(intent)
        cat_features = [1.0 if cat == c else 0.0 for c in ActionCategory]

        # Combine into a dict in the order of feature_names
        feature_dict = {
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'env_prod': env_prod,
            'user_role': user_role,
            'cat_database': 1.0 if cat == ActionCategory.DATABASE else 0.0,
            'cat_network': 1.0 if cat == ActionCategory.NETWORK else 0.0,
            'cat_compute': 1.0 if cat == ActionCategory.COMPUTE else 0.0,
            'cat_security': 1.0 if cat == ActionCategory.SECURITY else 0.0,
            'cat_default': 1.0 if cat == ActionCategory.DEFAULT else 0.0,
        }

        # Build feature vector in the order of self.feature_names
        x = np.array([feature_dict.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)

        if self.feature_scaler:
            x = self.feature_scaler.transform(x)

        # Linear predictor: intercept + sum(beta_i * x_i)
        # We stored coefficients with names like 'alpha', 'beta_sin_hour', etc.
        lin = self.coefficients.get('alpha', 0.0)
        for i, name in enumerate(self.feature_names):
            beta_name = f'beta_{name}'
            if beta_name in self.coefficients:
                lin += self.coefficients[beta_name] * x[0, i]

        # Logistic transform
        prob = 1.0 / (1.0 + np.exp(-lin))
        return float(prob)

    def _categorize_intent(self, intent: InfrastructureIntent) -> ActionCategory:
        """Helper to categorize intent (duplicate of later function)."""
        if isinstance(intent, ProvisionResourceIntent):
            if intent.resource_type in (ResourceType.DATABASE, ResourceType.STORAGE_ACCOUNT):
                return ActionCategory.DATABASE
            elif intent.resource_type == ResourceType.VM:
                return ActionCategory.COMPUTE
            elif intent.resource_type == ResourceType.VIRTUAL_NETWORK:
                return ActionCategory.NETWORK
        elif isinstance(intent, GrantAccessIntent):
            return ActionCategory.SECURITY
        elif isinstance(intent, DeployConfigurationIntent):
            if "database" in intent.service_name.lower():
                return ActionCategory.DATABASE
            return ActionCategory.COMPUTE
        return ActionCategory.DEFAULT

    def train(self, incidents_df: pd.DataFrame):
        """
        Train the HMC model on historical incident data.
        incidents_df must contain columns: outcome (0/1), hour, env_prod, user_role, category, etc.
        """
        # Prepare features
        # Create sin_hour, cos_hour
        incidents_df['sin_hour'] = np.sin(2 * np.pi * incidents_df['hour'] / 24)
        incidents_df['cos_hour'] = np.cos(2 * np.pi * incidents_df['hour'] / 24)

        # One‑hot encode category
        category_dummies = pd.get_dummies(incidents_df['category'], prefix='cat')
        feature_cols = ['sin_hour', 'cos_hour', 'env_prod', 'user_role'] + list(category_dummies.columns)
        X = incidents_df[feature_cols].values.astype(float)
        y = incidents_df['outcome'].values.astype(int)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build PyMC model
        n_features = X_scaled.shape[1]
        with pm.Model() as hmc_model:
            alpha = pm.Normal('alpha', mu=0, sigma=2)
            beta = pm.Normal('beta', mu=0, sigma=2, shape=n_features)
            mu = alpha + pm.math.dot(X_scaled, beta)
            pm.Bernoulli('y', logit_p=mu, observed=y)

            # Sample using NUTS
            trace = pm.sample(draws=1000, tune=1000, chains=2, step=pm.NUTS(), progressbar=False)

        # Save posterior means as coefficients
        self.coefficients = {}
        self.coefficients['alpha'] = float(trace.posterior['alpha'].mean())
        for i, name in enumerate(feature_cols):
            self.coefficients[f'beta_{name}'] = float(trace.posterior['beta'][..., i].mean())
        self.feature_names = feature_cols
        self.feature_scaler = scaler
        self.is_ready = True
        self._save(trace, feature_cols, scaler)
        logger.info("HMC model training completed.")


# =============================================================================
# Helper: categorize intent (public version)
# =============================================================================
def categorize_intent(intent: InfrastructureIntent) -> ActionCategory:
    if isinstance(intent, ProvisionResourceIntent):
        if intent.resource_type in (ResourceType.DATABASE, ResourceType.STORAGE_ACCOUNT):
            return ActionCategory.DATABASE
        elif intent.resource_type == ResourceType.VM:
            return ActionCategory.COMPUTE
        elif intent.resource_type == ResourceType.VIRTUAL_NETWORK:
            return ActionCategory.NETWORK
    elif isinstance(intent, GrantAccessIntent):
        return ActionCategory.SECURITY
    elif isinstance(intent, DeployConfigurationIntent):
        if "database" in intent.service_name.lower():
            return ActionCategory.DATABASE
        return ActionCategory.COMPUTE
    return ActionCategory.DEFAULT


# =============================================================================
# Combined Risk Engine
# =============================================================================
class RiskEngine:
    """
    Bayesian risk engine that combines online conjugate priors and offline HMC.

    The final risk score is a weighted average:
        risk = w * conjugate_mean + (1 - w) * hmc_prediction
    where w decreases as the HMC model has more training data (or as the context
    is well‑represented). A simple heuristic: w = max(0, 1 - n_data / N0),
    with N0 a threshold (e.g., 1000 incidents). In the absence of HMC, w=1.
    """

    def __init__(self, hmc_model_path: str = "hmc_model.json", n0: int = 1000):
        self.beta_store = BetaStore()
        self.hmc_model = HMCModel(hmc_model_path)
        self.n0 = n0  # threshold for HMC confidence
        self.total_incidents = 0  # could be loaded from a persistent counter
        self._lock = threading.RLock()

    def calculate_risk(
        self,
        intent: InfrastructureIntent,
        cost_estimate: Optional[float],
        policy_violations: List[str],
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Compute combined Bayesian risk score.
        """
        category = categorize_intent(intent)
        alpha, beta = self.beta_store.get(category)
        conjugate_risk = alpha / (alpha + beta)

        # HMC prediction (if available)
        hmc_risk = self.hmc_model.predict(intent, {"cost": cost_estimate, "violations": policy_violations})
        if hmc_risk is None:
            # HMC not ready – use only conjugate
            final_risk = conjugate_risk
            weight_hmc = 0.0
        else:
            with self._lock:
                n = self.total_incidents
            weight_hmc = min(1.0, n / self.n0)
            final_risk = (1 - weight_hmc) * conjugate_risk + weight_hmc * hmc_risk

        # Context multiplier (from article)
        multiplier = self._context_multiplier(intent)
        final_risk = min(final_risk * multiplier, 1.0)

        # Build explanation
        explanation = (
            f"Bayesian risk for category '{category.value}': "
            f"conjugate mean = {conjugate_risk:.3f} (α={alpha:.1f}, β={beta:.1f}). "
            f"HMC contribution: weight={weight_hmc:.2f}, prediction={hmc_risk if hmc_risk else 'N/A'}. "
            f"Context multiplier: {multiplier:.2f}. Final risk: {final_risk:.3f}."
        )

        contributions = {
            "conjugate_mean": conjugate_risk,
            "conjugate_alpha": alpha,
            "conjugate_beta": beta,
            "hmc_prediction": hmc_risk if hmc_risk is not None else 0.0,
            "hmc_weight": weight_hmc,
            "context_multiplier": multiplier,
        }

        return final_risk, explanation, contributions

    def update_outcome(self, intent: InfrastructureIntent, success: bool) -> None:
        """
        Update the conjugate prior with the outcome of an executed intent.
        Also increment total incident count for HMC weight calculation.
        """
        category = categorize_intent(intent)
        self.beta_store.update(category, success)
        with self._lock:
            self.total_incidents += 1

    def train_hmc(self, incidents_df: pd.DataFrame) -> None:
        """
        Train the HMC model on historical incident data.
        This should be called offline (e.g., via a background job).
        """
        self.hmc_model.train(incidents_df)

    def _context_multiplier(self, intent: InfrastructureIntent) -> float:
        """Compute multiplier based on environment, user role, time, etc."""
        mult = 1.0
        if hasattr(intent, "environment") and intent.environment == Environment.PROD:
            mult *= 1.5
        # Additional factors could be added
        return mult


# For backward compatibility, we also export RiskFactor (dummy)
class RiskFactor:
    """Placeholder for compatibility – not used."""
    def __init__(self, *args, **kwargs):
        pass
