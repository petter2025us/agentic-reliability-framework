import pytest
from agentic_reliability_framework.infrastructure.azure.azure_simulator import AzureInfrastructureSimulator
from agentic_reliability_framework.infrastructure.policies import Policy
from agentic_reliability_framework.infrastructure.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
)
from agentic_reliability_framework.infrastructure.healing_intent import RecommendedAction


def test_simulator_approve_low_risk():
    policies = [Policy(name="allow all", description="no restrictions")]
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="dev"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.APPROVE
    assert result.risk_score < 0.4
    assert result.cost_projection == 70.0


def test_simulator_deny_high_risk():
    policies = [Policy(name="no admin", max_permission_level="write")]
    simulator = AzureInfrastructureSimulator(policies)
    intent = GrantAccessIntent(
        principal="bob",
        permission_level="admin",
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.DENY
    assert result.risk_score > 0.8
    assert "admin" in result.justification


def test_simulator_escalate_medium_risk():
    policies = [Policy(name="allow all", description="no restrictions")]
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D16s_v3",  # high cost, but no policy violations
        requester="alice",
        environment="prod"
    )
    result = simulator.evaluate(intent)
    # risk around 0.13 -> still low, but we want to test escalate case.
    # To force escalate, we need a scenario with risk between 0.4 and 0.8.
    # Let's craft a medium risk: grant access with write permission (0.3+0.4*0.3=0.42)
    intent2 = GrantAccessIntent(
        principal="bob",
        permission_level="write",
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    result2 = simulator.evaluate(intent2)
    assert result2.recommended_action == RecommendedAction.ESCALATE
    assert 0.4 < result2.risk_score < 0.8
