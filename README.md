# ARF Infrastructure Governance Module (OSS)

This module provides an **advisory** simulation engine for evaluating infrastructure changes against policies, cost, and risk. It is part of the open-source Agentic Reliability Framework and is designed to be extended by the enterprise layer for actual enforcement.

## Purpose
- Simulate Azure resource provisioning, configuration deployments, and access grants.
- Estimate costs using built-in pricing tables.
- Evaluate against configurable policies.
- Compute a risk score (0–1).
- Output a `HealingIntent` with a recommended action: approve, deny, or escalate.

## Architecture
- **Intents** – Pydantic models representing infrastructure change requests.
- **Policies** – Configuration-driven rules (region restrictions, cost limits, permission bounds, etc.).
- **CostEstimator** – Deterministic cost estimation based on static pricing.
- **RiskEngine** – Combines intent type, cost, permissions, and policy violations into a risk score.
- **AzureInfrastructureSimulator** – Orchestrates the evaluation and returns a `HealingIntent`.

## Usage Example
```python
from agentic_reliability_framework.infrastructure import (
    AzureInfrastructureSimulator,
    Policy,
    ProvisionResourceIntent,
    ResourceType,
)

# Define policies
policies = [
    Policy(name="cost control", max_cost_threshold_usd=500.0),
    Policy(name="region allowlist", allowed_regions=["eastus", "westeurope"]),
]

# Create simulator
sim = AzureInfrastructureSimulator(policies)

# Create an intent
intent = ProvisionResourceIntent(
    resource_type=ResourceType.VM,
    region="eastus",
    size="Standard_D8s_v3",
    requester="alice",
    environment="prod"
)

# Evaluate
result = sim.evaluate(intent)
print(result.recommended_action)  # e.g., "deny"
print(result.justification)
