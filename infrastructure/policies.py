"""
Policy definitions and evaluation engine.

Policies are configuration-driven rules that infrastructure intents must satisfy.
The evaluator checks an intent against a list of policies and returns violations.
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
    # Cost threshold (USD per month) – applies to ProvisionResourceIntent
    max_cost_threshold_usd: Optional[float] = None
    # Allowed Azure regions – e.g., ["eastus", "westeurope"]
    allowed_regions: Optional[List[str]] = None
    # Maximum blast radius: "resource-group", "subscription", "management-group"
    blast_radius_limit: Optional[str] = None
    # Resource types that are forbidden (e.g., ["kubernetes_cluster"])
    restricted_resource_types: Optional[List[str]] = None
    # Maximum permission level: "read", "write", "admin"
    max_permission_level: Optional[str] = None


class PolicyEvaluator:
    """Evaluates an intent against a set of policies."""

    def __init__(self, policies: List[Policy]):
        self.policies = policies

    def evaluate(self, intent: InfrastructureIntent) -> List[str]:
        """
        Return a list of policy violation descriptions.
        If the intent satisfies all policies, the list is empty.
        """
        violations = []

        for policy in self.policies:
            # Check cost threshold for provisioning intents
            if (
                isinstance(intent, ProvisionResourceIntent)
                and policy.max_cost_threshold_usd is not None
            ):
                # Note: cost is not known here; we'll rely on the cost estimator later.
                # For policy evaluation we simply note that a cost policy exists;
                # actual violation is handled in the simulator after cost estimation.
                # To keep policy evaluation pure, we don't compute cost here.
                # We'll instead pass a flag that cost will be checked separately.
                # Alternatively, we could pass an estimated cost, but that couples.
                # We'll keep policy evaluation simple – only static constraints.
                pass

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
                if level_order.get(intent.permission_level, 0) > level_order.get(policy.max_permission_level, 0):
                    violations.append(
                        f"Policy '{policy.name}': permission level '{intent.permission_level}' exceeds "
                        f"maximum allowed '{policy.max_permission_level}'."
                    )

            # Blast radius checks (simplified: if scope is larger than allowed)
            if (
                policy.blast_radius_limit is not None
            ):
                # For provision intents, we could check resource scope against limit
                # For simplicity, we skip detailed blast radius in this version.
                pass

        return violations
