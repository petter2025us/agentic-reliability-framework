"""
Risk scoring engine with configurable weights.

Calculates a risk score between 0 and 1 based on intent type, cost, permissions,
deployment scope, and policy violations. Weights can be customized.
"""

from typing import List, Dict, Any, Tuple, Optional
from agentic_reliability_framework.infrastructure.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    PermissionLevel,
    Environment,
)


class RiskEngine:
    """Calculates risk score and explanation for an infrastructure intent."""

    DEFAULT_WEIGHTS = {
        "intent_type": {
            "provision_resource": 0.1,
            "grant_access": 0.3,
            "deploy_config": 0.2,
        },
        "cost": 0.3,                # weight applied to normalised cost factor
        "permission_level": 0.3,    # weight applied to permission factor
        "deployment_scope": 0.2,    # weight applied to scope factor
        "policy_violation": 0.2,    # weight per violation (capped at 0.8)
        "environment": 0.1,          # extra risk for prod
    }

    def __init__(self, weights: Optional[Dict[str, Any]] = None):
        """
        Initialize with optional custom weights.
        Weights not provided will use the defaults.
        """
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

    def calculate_risk(
        self,
        intent: InfrastructureIntent,
        cost_estimate: Optional[float],
        policy_violations: List[str],
    ) -> Tuple[float, str]:
        """
        Return a tuple (risk_score, explanation).
        risk_score is a float in [0, 1].
        """
        factors = []  # list of (factor_name, contribution, explanation_part)

        # 1. Intent type base risk
        intent_key = intent.intent_type
        base_risk = self.weights["intent_type"].get(intent_key, 0.1)
        factors.append(("intent_type", base_risk, f"intent type '{intent_key}'"))

        # 2. Cost factor (if applicable)
        if cost_estimate is not None and isinstance(intent, ProvisionResourceIntent):
            # Normalise cost: $0 → 0, $5000 → 1 (linear)
            cost_factor = min(cost_estimate / 5000.0, 1.0)
            weighted_cost = cost_factor * self.weights["cost"]
            factors.append(("cost", weighted_cost, f"estimated cost ${cost_estimate:.2f}"))
        else:
            weighted_cost = 0.0

        # 3. Permission level (for access grants)
        perm_contribution = 0.0
        perm_expl = ""
        if isinstance(intent, GrantAccessIntent):
            perm_map = {PermissionLevel.READ: 0.1, PermissionLevel.WRITE: 0.4, PermissionLevel.ADMIN: 0.8}
            perm_factor = perm_map.get(intent.permission_level, 0.5)
            perm_contribution = perm_factor * self.weights["permission_level"]
            factors.append(("permission", perm_contribution, f"permission level '{intent.permission_level.value}'"))

        # 4. Deployment scope (for config changes)
        scope_contribution = 0.0
        if isinstance(intent, DeployConfigurationIntent):
            scope_map = {"single_instance": 0.1, "canary": 0.2, "global": 0.6}
            scope_factor = scope_map.get(intent.change_scope, 0.3)
            scope_contribution = scope_factor * self.weights["deployment_scope"]
            factors.append(("scope", scope_contribution, f"deployment scope '{intent.change_scope}'"))

        # 5. Environment (prod adds risk)
        env_contribution = 0.0
        if hasattr(intent, "environment") and intent.environment == Environment.PROD:
            env_contribution = self.weights["environment"]
            factors.append(("environment", env_contribution, "production environment"))

        # 6. Policy violations
        violation_contribution = min(len(policy_violations) * self.weights["policy_violation"], 0.8)
        if violation_contribution > 0:
            factors.append(("policy_violations", violation_contribution, f"{len(policy_violations)} violation(s)"))

        # Sum contributions
        total_score = sum(contrib for _, contrib, _ in factors)
        # Cap at 1.0
        total_score = min(total_score, 1.0)

        # Build explanation
        explanation_parts = ["Risk contributions:"] + [f"- {exp}: {contrib:.2f}" for _, contrib, exp in factors]
        explanation = "\n".join(explanation_parts)

        return round(total_score, 2), explanation
