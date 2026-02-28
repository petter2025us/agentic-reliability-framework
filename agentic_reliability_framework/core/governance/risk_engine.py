"""
Bayesian Risk Engine – Learns risk factor weights from historical data using HMC.
Provides probabilistic risk scores with full uncertainty quantification.
"""

import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    PermissionLevel,
    Environment,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """Definition of a risk factor with a function to extract its value from an intent."""
    name: str
    extractor: Callable[[InfrastructureIntent], float]
    description: str = ""


class BayesianRiskEngine:
    """
    Bayesian logistic regression model for risk scoring.

    The model learns weights for each risk factor from historical data,
    producing a posterior distribution over weights. For a new intent,
    it outputs a full distribution of the risk probability.
    """

    def __init__(self, factors: Optional[List[RiskFactor]] = None):
        if factors is None:
            # Default factors (matching original risk_engine)
            self.factors = [
                RiskFactor("intent_type", self._extract_intent_type, "Base risk from intent type"),
                RiskFactor("cost", self._extract_cost, "Normalized cost estimate"),
                RiskFactor("permission", self._extract_permission, "Permission level being granted"),
                RiskFactor("scope", self._extract_scope, "Deployment scope"),
                RiskFactor("environment", self._extract_environment, "Production environment"),
                RiskFactor("policy_violations", self._extract_policy_violations, "Number of policy violations"),
            ]
        else:
            self.factors = factors

        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.is_trained = False
        self.feature_stats: Dict[str, Tuple[float, float]] = {}  # (mean, std) for scaling

    # ---------- Factor extraction methods ----------
    def _extract_intent_type(self, intent: InfrastructureIntent) -> float:
        mapping = {
            "provision_resource": 0.1,
            "grant_access": 0.3,
            "deploy_config": 0.2,
        }
        return mapping.get(intent.intent_type, 0.1)

    def _extract_cost(self, intent: InfrastructureIntent) -> float:
        if isinstance(intent, ProvisionResourceIntent):
            # Cost is not part of intent; it must be provided separately.
            # For factor extraction, we'll rely on a separate cost input passed in context.
            # We'll treat cost as an extra feature; we'll handle it in _prepare_features.
            return 0.0  # Placeholder; actual cost will be added in training data.
        return 0.0

    def _extract_permission(self, intent: InfrastructureIntent) -> float:
        if isinstance(intent, GrantAccessIntent):
            mapping = {
                PermissionLevel.READ: 0.1,
                PermissionLevel.WRITE: 0.4,
                PermissionLevel.ADMIN: 0.8,
            }
            return mapping.get(intent.permission_level, 0.5)
        return 0.0

    def _extract_scope(self, intent: InfrastructureIntent) -> float:
        if isinstance(intent, DeployConfigurationIntent):
            mapping = {
                "single_instance": 0.1,
                "canary": 0.2,
                "global": 0.6,
            }
            return mapping.get(intent.change_scope, 0.3)
        return 0.0

    def _extract_environment(self, intent: InfrastructureIntent) -> float:
        if hasattr(intent, "environment") and intent.environment == Environment.PROD:
            return 1.0
        return 0.0

    def _extract_policy_violations(self, intent: InfrastructureIntent) -> float:
        # This is a count; we'll use it directly. The extractor itself can't access violations;
        # they must be provided in context.
        return 0.0  # Placeholder

    # ---------- Feature preparation ----------
    def _prepare_features(self, historical_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert list of historical incidents to feature matrix and binary target.
        Each dict must contain:
            - 'intent': InfrastructureIntent
            - 'cost' (optional)
            - 'policy_violations' (list)
            - 'outcome': 1 if incident was critical/high, else 0
        """
        rows = []
        targets = []
        for record in historical_data:
            intent = record['intent']
            # Extract factor values
            feature_vec = []
            for factor in self.factors:
                val = factor.extractor(intent)
                # If factor requires context, add from record
                if factor.name == 'cost':
                    val = record.get('cost', 0.0)
                elif factor.name == 'policy_violations':
                    val = len(record.get('policy_violations', [])) * 0.2  # each violation adds 0.2
                feature_vec.append(val)
            rows.append(feature_vec)
            targets.append(record['outcome'])

        X = np.array(rows, dtype=float)
        y = np.array(targets, dtype=int)

        # Scale features for better sampling
        self.feature_stats = {}
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[1]):
            mean = X[:, i].mean()
            std = X[:, i].std()
            if std == 0:
                std = 1.0
            self.feature_stats[self.factors[i].name] = (mean, std)
            X_scaled[:, i] = (X[:, i] - mean) / std

        return X_scaled, y

    # ---------- Model building ----------
    def build_model(self, X: np.ndarray, y: np.ndarray):
        """Build Bayesian logistic regression model."""
        n_features = X.shape[1]
        with pm.Model() as self.model:
            # Priors for coefficients (regularizing)
            alpha = pm.Normal('alpha', mu=0, sigma=2)  # intercept
            beta = pm.Normal('beta', mu=0, sigma=2, shape=n_features)

            # Linear predictor
            mu = alpha + pm.math.dot(X, beta)

            # Likelihood
            pm.Bernoulli('y', logit_p=mu, observed=y)

            # Compute risk probability for new predictions later
            # We'll handle predictions separately.

        logger.info(f"Built Bayesian logistic regression model with {n_features} features.")

    # ---------- Training ----------
    def train(self, historical_data: List[Dict[str, Any]],
              draws: int = 1000, tune: int = 1000, chains: int = 2,
              target_accept: float = 0.9):
        """
        Train the model on historical data using NUTS.
        """
        if len(historical_data) < 10:
            logger.warning("Insufficient data for training (need at least 10 records).")
            self.is_trained = False
            return

        X, y = self._prepare_features(historical_data)
        self.build_model(X, y)

        with self.model:
            logger.info("Sampling with NUTS (target_accept=%.2f)...", target_accept)
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                step=pm.NUTS(target_accept=target_accept),
                progressbar=False,
                return_inferencedata=True,
                idata_kwargs={'log_likelihood': True}
            )
            self.is_trained = True
            logger.info("NUTS sampling completed.")

    # ---------- Prediction ----------
    def predict_risk(self, intent: InfrastructureIntent,
                     cost: Optional[float] = None,
                     policy_violations: Optional[List[str]] = None,
                     n_samples: int = 1000) -> Dict[str, Any]:
        """
        Predict risk probability distribution for a new intent.

        Returns:
            Dictionary with:
                - mean_risk: float
                - std_risk: float
                - p5, p50, p95: percentiles
                - samples: list of risk samples
                - factor_contributions: dict of weighted contributions (deterministic using mean weights)
        """
        if not self.is_trained or self.trace is None:
            logger.warning("Model not trained. Returning default risk.")
            return {
                'mean_risk': 0.5,
                'std_risk': 0.0,
                'p5': 0.5,
                'p50': 0.5,
                'p95': 0.5,
                'samples': [0.5],
                'factor_contributions': {}
            }

        # Build feature vector for new intent
        feature_vec = []
        for factor in self.factors:
            val = factor.extractor(intent)
            if factor.name == 'cost' and cost is not None:
                val = cost
            elif factor.name == 'policy_violations' and policy_violations is not None:
                val = len(policy_violations) * 0.2
            feature_vec.append(val)

        # Scale using training stats
        X_new = np.array(feature_vec).reshape(1, -1)
        for i, factor in enumerate(self.factors):
            mean, std = self.feature_stats.get(factor.name, (0.0, 1.0))
            X_new[0, i] = (X_new[0, i] - mean) / std

        # Extract posterior samples
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, len(self.factors))

        # Compute linear predictor for each sample
        mu = alpha_samples + np.dot(beta_samples, X_new.T).flatten()
        risk_probs = 1 / (1 + np.exp(-mu))

        # Compute factor contributions (using mean weights for explanation)
        mean_beta = beta_samples.mean(axis=0)
        contributions = {}
        for i, factor in enumerate(self.factors):
            # Contribution = beta_i * x_i (unscaled) – but we need to interpret. For explanation,
            # we can compute the effect on log-odds.
            # Simpler: return mean weight times feature value (after scaling?).
            # For now, we'll return the mean weight as importance.
            contributions[factor.name] = float(mean_beta[i])

        return {
            'mean_risk': float(np.mean(risk_probs)),
            'std_risk': float(np.std(risk_probs)),
            'p5': float(np.percentile(risk_probs, 5)),
            'p50': float(np.percentile(risk_probs, 50)),
            'p95': float(np.percentile(risk_probs, 95)),
            'samples': risk_probs.tolist()[:100],  # first 100 samples
            'factor_contributions': contributions
        }

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Return posterior summary of coefficients for each factor.
        """
        if not self.is_trained or self.trace is None:
            return {}
        result = {}
        for i, factor in enumerate(self.factors):
            beta_samples = self.trace.posterior['beta'].values[..., i].flatten()
            result[factor.name] = {
                'mean': float(np.mean(beta_samples)),
                'std': float(np.std(beta_samples)),
                'p5': float(np.percentile(beta_samples, 5)),
                'p95': float(np.percentile(beta_samples, 95)),
                'p_gt_0': float((beta_samples > 0).mean())
            }
        return result
