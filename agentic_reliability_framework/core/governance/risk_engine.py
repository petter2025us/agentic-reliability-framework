# agentic_reliability_framework/infrastructure/risk_engine.py
"""
Risk Scoring Engine – Multi-factor probabilistic risk model.

This module computes a risk score (0-1) for an infrastructure intent by combining
multiple factors with configurable weights. The model is inspired by Bayesian
decision theory and multi-criteria decision analysis (MCDA). It produces not only
a score but also a detailed explanation of each factor's contribution, supporting
transparency and psychological trust.

The risk engine is designed to be extended with additional factors (e.g., historical
data, anomaly scores) without changing the core API.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    PermissionLevel,
    Environment,
)

# -----------------------------------------------------------------------------
# Factor Definition
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskFactor:
    """A single factor contributing to risk, with a weight and a scoring function."""
    name: str
    weight: float
    score_fn: Callable[[InfrastructureIntent, Optional[float], List[str]], float]
    description: str = ""

    def __call__(self, intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
        return self.score_fn(intent, cost, violations)

# -----------------------------------------------------------------------------
# Built-in Factors
# -----------------------------------------------------------------------------
def intent_type_factor(intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
    """Base risk from intent type."""
    mapping = {
        "provision_resource": 0.1,
        "grant_access": 0.3,
        "deploy_config": 0.2,
    }
    return mapping.get(intent.intent_type, 0.1)

def cost_factor(intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
    """Risk contribution from estimated cost (normalized to [0,1])."""
    if not isinstance(intent, ProvisionResourceIntent) or cost is None:
        return 0.0
    # Normalize: $0 → 0, $5000 → 1 (linear)
    return min(cost / 5000.0, 1.0)

def permission_factor(intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
    """Risk from permission level being granted."""
    if not isinstance(intent, GrantAccessIntent):
        return 0.0
    mapping = {
        PermissionLevel.READ: 0.1,
        PermissionLevel.WRITE: 0.4,
        PermissionLevel.ADMIN: 0.8,
    }
    return mapping.get(intent.permission_level, 0.5)

def scope_factor(intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
    """Risk from deployment scope (for config changes)."""
    if not isinstance(intent, DeployConfigurationIntent):
        return 0.0
    mapping = {
        "single_instance": 0.1,
        "canary": 0.2,
        "global": 0.6,
    }
    return mapping.get(intent.change_scope, 0.3)

def environment_factor(intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
    """Additional risk if environment is production."""
    if hasattr(intent, "environment") and intent.environment == Environment.PROD:
        return 0.1
    return 0.0

def policy_violation_factor(intent: InfrastructureIntent, cost: Optional[float], violations: List[str]) -> float:
    """Risk from number of policy violations (capped)."""
    # Each violation adds 0.2, max 0.8
    return min(len(violations) * 0.2, 0.8)

# -----------------------------------------------------------------------------
# Risk Engine
# -----------------------------------------------------------------------------
class RiskEngine:
    """
    Computes a weighted risk score from multiple factors.

    The engine is initialized with a list of factors and their weights.
    The total score is the weighted sum of factor scores, clamped to [0,1].
    """

    DEFAULT_FACTORS = [
        RiskFactor("intent_type", 1.0, intent_type_factor, "Base risk from intent type"),
        RiskFactor("cost", 0.3, cost_factor, "Normalized cost estimate"),
        RiskFactor("permission", 0.3, permission_factor, "Permission level being granted"),
        RiskFactor("scope", 0.2, scope_factor, "Deployment scope"),
        RiskFactor("environment", 0.1, environment_factor, "Production environment"),
        RiskFactor("policy_violations", 0.2, policy_violation_factor, "Number of policy violations"),
    ]

    def __init__(self, factors: Optional[List[RiskFactor]] = None):
        """
        Initialize with custom factors. If none provided, uses DEFAULT_FACTORS.
        """
        self.factors = factors if factors is not None else self.DEFAULT_FACTORS

    def calculate_risk(
        self,
        intent: InfrastructureIntent,
        cost_estimate: Optional[float],
        policy_violations: List[str],
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Compute risk score and detailed breakdown.

        Returns:
            - total_score: float in [0,1]
            - explanation: human-readable string
            - contributions: dict mapping factor names to their weighted contribution
        """
        total = 0.0
        contributions = {}

        for factor in self.factors:
            raw_score = factor(intent, cost_estimate, policy_violations)
            weighted = raw_score * factor.weight
            contributions[factor.name] = weighted
            total += weighted

        # Clamp to [0,1]
        total = max(0.0, min(total, 1.0))

        # Build explanation
        lines = [f"Total risk score: {total:.2f}"]
        for factor in self.factors:
            contrib = contributions[factor.name]
            if contrib > 0.0:
                lines.append(f"  - {factor.name}: {contrib:.2f} ({factor.description})")
        explanation = "\n".join(lines)

        return total, explanation, contributions
