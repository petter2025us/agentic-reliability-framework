# ARF Infrastructure Governance Module (OSS)

## Overview

This module provides a **deterministic, advisory** simulation engine for evaluating infrastructure changes against policies, cost, and risk. It is the open-source core of the Agentic Reliability Framework, designed to be extended by a closed-source enterprise layer that enforces the decisions.

The design emphasizes:

- **Mathematical elegance**: Algebraic data types for intents, composable policy algebra, probabilistic risk models.
- **Explainable AI (XAI)**: Detailed justification and factor breakdowns build trust.
- **Knowledge engineering**: Semantic validation using ontologies, provenance tracking.
- **Platform engineering**: Clean separation of concerns, dependency injection, configuration-driven components.
- **Decision engineering**: Weighted multi-factor risk scoring with configurable thresholds.

## Architecture

graph TB
    subgraph "Input Layer"
        A1[Agent/User Request]
        A2[Intent Factory]
        A3[Provenance Tracker]
    end
    
    subgraph "Intent Layer"
        B1[ProvisionResourceIntent]
        B2[DeployConfigurationIntent]
        B3[GrantAccessIntent]
        B4[Base Intent<br/>- intent_id<br/>- timestamp<br/>- requester<br/>- provenance]
    end
    
    subgraph "Policy Engine"
        C1[Policy Algebra<br/>AND/OR/NOT]
        C2[Atomic Policies]
        C3[Policy Evaluator]
        C4[Context-Aware<br/>Evaluation]
    end
    
    subgraph "Cost Engine"
        D1[Deterministic<br/>Pricing]
        D2[Bayesian<br/>Extension]
        D3[Cache Layer]
    end
    
    subgraph "Risk Engine"
        E1[Factor Registry]
        E2[Weighted Scoring]
        E3[Explanation Generator]
        E4[Contribution Analysis]
    end
    
    subgraph "Output Layer"
        F1[HealingIntent]
        F2[Recommended Action]
        F3[Justification]
        F4[Evaluation Details]
    end
    
    subgraph "Enterprise Extension"
        G1[Execution Gateway]
        G2[Azure API Proxy]
        G3[Audit Logger]
        G4[Compliance Reporter]
    end
    
    A1 --> A2
    A2 --> A3
    A3 --> B1 & B2 & B3
    
    B1 & B2 & B3 --> C3
    C1 --> C2
    C2 --> C3
    C3 --> E2
    
    B1 --> D1
    D1 --> D3
    D3 --> E2
    
    E1 --> E2
    E2 --> E3
    E3 --> E4
    
    E4 --> F1
    F1 --> F2 & F3 & F4
    
    F1 -.-> G1
    G1 --> G2
    G2 --> G3
    G3 --> G4
```

## Key Components

- **intents.py**: Discriminated union of infrastructure change requests (provision, deploy config, grant access).
- **policies.py**: Composable policy algebra with AND, OR, NOT combinators. Policies are evaluated against intents, returning violations.
- **cost_estimator.py**: Deterministic cost estimation from built-in or custom pricing tables.
- **risk_engine.py**: Multi-factor risk scoring with configurable weights and detailed explanations.
- **azure_simulator.py**: Orchestrates the evaluation, producing a `HealingIntent`.

## Usage Example

```python
from agentic_reliability_framework.infrastructure import (
    AzureInfrastructureSimulator,
    RegionAllowedPolicy,
    CostThresholdPolicy,
    ProvisionResourceIntent,
    ResourceType,
    Environment,
)

# Build a policy: region must be 'eastus' AND cost < 200
policy = RegionAllowedPolicy({"eastus"}) & CostThresholdPolicy(200.0)

sim = AzureInfrastructureSimulator(policy)

intent = ProvisionResourceIntent(
    resource_type=ResourceType.VM,
    region="westus",
    size="Standard_D8s_v3",
    requester="alice",
    environment=Environment.PROD
)

result = sim.evaluate(intent)
print(result.recommended_action)  # "deny"
print(result.justification)
```
