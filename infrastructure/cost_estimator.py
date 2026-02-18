"""
Simulated cost estimation for Azure resources.

Uses built-in pricing tables to compute monthly costs deterministically.
No external API calls.
"""

from typing import Optional

from agentic_reliability_framework.infrastructure.intents import ProvisionResourceIntent, ResourceType


class CostEstimator:
    """Deterministic cost estimator based on static pricing data."""

    def __init__(self):
        # Simulated monthly prices in USD.
        # In a real implementation, these could be loaded from a config file.
        self.pricing = {
            ResourceType.VM: {
                "Standard_D2s_v3": 70.0,
                "Standard_D4s_v3": 140.0,
                "Standard_D8s_v3": 280.0,
                "Standard_D16s_v3": 560.0,
            },
            ResourceType.STORAGE_ACCOUNT: {
                "50GB": 5.0,
                "100GB": 10.0,
                "1TB": 100.0,
                "10TB": 900.0,
            },
            ResourceType.DATABASE: {
                "Basic": 15.0,
                "Standard": 50.0,
                "Premium": 200.0,
            },
            ResourceType.KUBERNETES_CLUSTER: {
                "Small": 100.0,
                "Medium": 300.0,
                "Large": 600.0,
            },
            ResourceType.FUNCTION_APP: {
                "Consumption": 0.0,  # pay-per-use, negligible baseline
                "Premium": 75.0,
            },
            ResourceType.VIRTUAL_NETWORK: {
                "default": 0.0,  # typically no direct cost
            },
        }

    def estimate_monthly_cost(self, intent: ProvisionResourceIntent) -> Optional[float]:
        """
        Return estimated monthly cost in USD for the given provision intent.
        Returns None if the size is unknown.
        """
        resource_pricing = self.pricing.get(intent.resource_type)
        if not resource_pricing:
            return None

        # Attempt to match size exactly, otherwise try a default
        if intent.size in resource_pricing:
            return resource_pricing[intent.size]
        # Fallback: if size looks like a known pattern, try to approximate
        # For simplicity, return None if exact match not found.
        return None

    def cost_delta_vs_baseline(
        self,
        intent: ProvisionResourceIntent,
        baseline_intent: Optional[ProvisionResourceIntent] = None,
    ) -> Optional[float]:
        """
        Compute the cost difference between the proposed intent and a baseline.
        If no baseline is provided, compare to a minimal "reasonable" configuration.
        Returns None if cost cannot be estimated for either.
        """
        proposed_cost = self.estimate_monthly_cost(intent)
        if proposed_cost is None:
            return None

        if baseline_intent:
            baseline_cost = self.estimate_monthly_cost(baseline_intent)
            if baseline_cost is None:
                return None
            return proposed_cost - baseline_cost
        else:
            # Default baseline: smallest available size for that resource type
            resource_pricing = self.pricing.get(intent.resource_type)
            if not resource_pricing:
                return None
            # Sort by value and take the smallest
            min_cost = min(resource_pricing.values())
            return proposed_cost - min_cost
