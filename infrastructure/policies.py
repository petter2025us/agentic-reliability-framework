"""
Policy definitions and evaluation engine.

Policies are configuration‑driven rules that infrastructure intents must satisfy.
Static checks (region, resource type, permission level) are evaluated here.
Dynamic checks (like cost thresholds) are handled in the simulator.
"""

from typing import List, Optional
from pydantic import BaseModel

from agentic_reliability_framework.infrastructure.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
)


class Policy(BaseModel):
    """A single policy rule with constraints."""
    name: str
    description: str
    # Cost threshold in USD per month – checked in simulator after cost estimation
    cost_threshold_usd: Optional[float] = None
    # Allowed Azure regions – e.g., ["eastus", "westeurope"]
    allowed_regions: Optional[List[str]] = None
    # Maximum blast radius: "resource-group", "subscription", "management-group"
    blast_radius_limit: Optional[str] = None
    # Resource types that are forbidden (e.g., ["kubernetes_cluster"])
    restricted_resource_types: Optional[List[str]] = None
    # Maximum permission level: "read", "write", "admin"
    max_permission_level: Optional[str] = None

    def check_cost(self, estimated_cost: Optional[float]) -> Optional[str]:
        """
        Return a violation string if the estimated cost exceeds the threshold.
        If no threshold or no cost, return None.
        """
        if self.cost_threshold_usd is not None and estimated_cost is not None:
            if estimated_cost > self.cost_threshold_usd:
                return (
                    f"Policy '{self.name}': estimated cost ${estimated_cost:.2f} "
                    f"exceeds threshold ${self.cost_threshold_usd:.2f}"
                )
        return None


class PolicyEvaluator:
    """Evaluates an intent against a set of policies (static checks only)."""

    def __init__(self, policies: List[Policy]):
        self.policies = policies

    def evaluate(self, intent: InfrastructureIntent) -> List[str]:
        """
        Return a list of policy violation descriptions from static rules.
        """
        violations = []

        for policy in self.policies:
            # Region restrictions
            if (
                isinstance(intent, ProvisionResourceIntent)
                and policy.allowed_regions is not None
                and intent.region not in policy.allowed_regions
            ):
                violations.append(
                    f"Policy '{policy.name}': region '{intent.region}' is not allowed. "
                    f"Allowed regions: {policy.allowed_regions}"
                )

            # Restricted resource types
            if (
                isinstance(intent, ProvisionResourceIntent)
                and policy.restricted_resource_types is not None
                and intent.resource_type.value in policy.restricted_resource_types
            ):
                violations.append(
                    f"Policy '{policy.name}': resource type '{intent.resource_type.value}' is restricted."
                )

            # Permission level limits
            if (
                isinstance(intent, GrantAccessIntent)
                and policy.max_permission_level is not None
            ):
                # Simple ordinal comparison: read < write < admin
                level_order = {"read": 1, "write": 2, "admin": 3}
                if level_order.get(intent.permission_level.value, 0) > level_order.get(policy.max_permission_level, 0):
                    violations.append(
                        f"Policy '{policy.name}': permission level '{intent.permission_level.value}' exceeds "
                        f"maximum allowed '{policy.max_permission_level}'."
                    )

            # Blast radius checks can be added later
        return violations
