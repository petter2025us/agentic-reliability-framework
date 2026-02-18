"""
Risk scoring engine.

Combines multiple factors (cost, permissions, policy violations) into a single
risk score between 0 (safe) and 1 (dangerous), with an explanation.
"""

from typing import List, Union

from agentic_reliability_framework.infrastructure.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
)


class RiskEngine:
    """Calculates risk score and explanation for an infrastructure intent."""

    def calculate_risk(
        self,
        intent: InfrastructureIntent,
        cost_estimate: Union[float, None],
        policy_violations: List[str],
    ) -> tuple[float, str]:
        """
        Return a tuple (risk_score, explanation).
        risk_score is a float in [0, 1].
        """
        score = 0.0
        factors = []

        # Base risk by intent type
        if isinstance(intent, ProvisionResourceIntent):
            factors.append(("intent type", 0.1))
        elif isinstance(intent, GrantAccessIntent):
            factors.append(("intent type", 0.3))  # access grants are riskier
        elif isinstance(intent, DeployConfigurationIntent):
            factors.append(("intent type", 0.2))

        # Cost factor (if applicable)
        if cost_estimate is not None:
            # Scale cost: $0 → 0, $1000 → 0.5, $5000 → 1.0 (linear cap at 5000)
            cost_score = min(cost_estimate / 5000.0, 1.0)
            factors.append(("cost", cost_score * 0.3))  # weight 30%

        # Permission level for access grants
        if isinstance(intent, GrantAccessIntent):
            perm_map = {"read": 0.1, "write": 0.4, "admin": 0.8}
            perm_score = perm_map.get(intent.permission_level, 0.5)
            factors.append(("permission level", perm_score * 0.3))

        # Deployment scope for config changes
        if isinstance(intent, DeployConfigurationIntent):
            scope_map = {"single_instance": 0.1, "canary": 0.2, "global": 0.6}
            scope_score = scope_map.get(intent.change_scope, 0.3)
            factors.append(("deployment scope", scope_score * 0.2))

        # Policy violations increase risk
        if policy_violations:
            # Each violation adds 0.2, capped at 0.8 total
            violation_penalty = min(len(policy_violations) * 0.2, 0.8)
            factors.append(("policy violations", violation_penalty))

        # Sum weighted factors
        score = sum(weight for _, weight in factors)

        # Cap at 1.0
        score = min(score, 1.0)

        # Build explanation
        explanation_parts = [f"Base risk from {intent.intent_type}."]
        if cost_estimate is not None:
            explanation_parts.append(f"Estimated cost ${cost_estimate:.2f}/month.")
        if policy_violations:
            explanation_parts.append(f"Policy violations: {', '.join(policy_violations)}.")
        explanation = " ".join(explanation_parts)

        return round(score, 2), explanation
