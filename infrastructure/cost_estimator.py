"""
Simulated cost estimation for Azure resources.

Pricing data can be loaded from a YAML file. A default built‑in dictionary is provided.
"""

import os
from typing import Dict, Any, Optional
from functools import lru_cache
import yaml

from agentic_reliability_framework.infrastructure.intents import ProvisionResourceIntent, ResourceType


class CostEstimator:
    """Deterministic cost estimator based on static pricing data."""

    DEFAULT_PRICING = {
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
            "Consumption": 0.0,
            "Premium": 75.0,
        },
        ResourceType.VIRTUAL_NETWORK: {
            "default": 0.0,
        },
    }

    def __init__(self, pricing_file: Optional[str] = None):
        """
        Initialize with pricing data.
        If pricing_file is given, load from YAML; otherwise use DEFAULT_PRICING.
        """
        if pricing_file and os.path.exists(pricing_file):
            with open(pricing_file, 'r') as f:
                raw = yaml.safe_load(f)
                # Convert resource type strings back to Enums
                self.pricing = {}
                for res_str, sizes in raw.items():
                    try:
                        res_type = ResourceType(res_str)
                    except ValueError:
                        # Skip unknown resource types
                        continue
                    self.pricing[res_type] = sizes
        else:
            self.pricing = self.DEFAULT_PRICING.copy()

    @lru_cache(maxsize=128)
    def estimate_monthly_cost(self, intent: ProvisionResourceIntent) -> Optional[float]:
        """
        Return estimated monthly cost in USD for the given provision intent.
        Returns None if the size is unknown.
        """
        resource_pricing = self.pricing.get(intent.resource_type)
        if not resource_pricing:
            return None

        # Attempt to match size exactly
        if intent.size in resource_pricing:
            return resource_pricing[intent.size]

        # Fallback: try to match case‑insensitively (if needed)
        # For simplicity, return None if not found
        return None

    def cost_delta_vs_baseline(
        self,
        intent: ProvisionResourceIntent,
        baseline_intent: Optional[ProvisionResourceIntent] = None,
    ) -> Optional[float]:
        """
        Compute the cost difference between the proposed intent and a baseline.
        If no baseline is provided, compare to the smallest available size for that resource type.
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
            # Default baseline: smallest size for that resource type
            resource_pricing = self.pricing.get(intent.resource_type)
            if not resource_pricing:
                return None
            # Find the minimum cost value
            min_cost = min(resource_pricing.values())
            return proposed_cost - min_cost
